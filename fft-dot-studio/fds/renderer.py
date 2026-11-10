"""ModernGL renderer: additive dot sprites into an HDR buffer, optional bloom, tonemapping, overlays."""
import numpy as np
import moderngl

DOT_VS = '''
#version 330
in vec2 in_pos;      // x in [-1,1], y in [0,1]
in float in_i;
in vec3 in_col;
uniform float u_size;
out float v_i;
out vec3 v_col;
void main() {
    gl_Position = vec4(in_pos.x, in_pos.y * 2.0 - 1.0, 0.0, 1.0);
    gl_PointSize = u_size;
    v_i = in_i;
    v_col = in_col;
}
'''
DOT_FS = '''
#version 330
in float v_i;
in vec3 v_col;
uniform int u_shape;   // 0 square, 1 round, 2 glow
out vec4 f_color;
void main() {
    float a = 1.0;
    if (u_shape != 0) {
        vec2 d = gl_PointCoord * 2.0 - 1.0;
        float r2 = dot(d, d);
        if (u_shape == 1) {
            a = clamp((1.0 - r2) * 3.0, 0.0, 1.0);
        } else {
            // core + halo; sprite is drawn 3x larger for the halo
            float core = clamp((1.0 / 9.0 - r2) * 27.0, 0.0, 1.0);
            a = core + 0.35 * exp(-r2 * 6.0);
        }
        if (a <= 0.0) discard;
    }
    f_color = vec4(v_col * v_i * a, 1.0);
}
'''
QUAD_VS = '''
#version 330
in vec2 in_vert;
out vec2 v_uv;
void main() { v_uv = in_vert * 0.5 + 0.5; gl_Position = vec4(in_vert, 0.0, 1.0); }
'''
BLUR_FS = '''
#version 330
uniform sampler2D tex;
uniform vec2 u_dir;
in vec2 v_uv;
out vec4 f_color;
void main() {
    float w[5] = float[](0.2270270, 0.1945946, 0.1216216, 0.0540540, 0.0162162);
    vec3 c = texture(tex, v_uv).rgb * w[0];
    for (int i = 1; i < 5; i++) {
        c += texture(tex, v_uv + u_dir * float(i)).rgb * w[i];
        c += texture(tex, v_uv - u_dir * float(i)).rgb * w[i];
    }
    f_color = vec4(c, 1.0);
}
'''
DOWN_FS = '''
#version 330
uniform sampler2D tex;
uniform vec2 u_texel;
in vec2 v_uv;
out vec4 f_color;
void main() {
    vec3 c = texture(tex, v_uv + u_texel * vec2(-0.5, -0.5)).rgb + texture(tex, v_uv + u_texel * vec2(0.5, -0.5)).rgb
           + texture(tex, v_uv + u_texel * vec2(-0.5, 0.5)).rgb + texture(tex, v_uv + u_texel * vec2(0.5, 0.5)).rgb;
    f_color = vec4(c * 0.25, 1.0);
}
'''
COMP_FS = '''
#version 330
uniform sampler2D accum;
uniform sampler2D bloom1;
uniform sampler2D bloom2;
uniform float u_glow;
uniform int u_tonemap;
uniform float u_sat;
uniform vec3 u_bg;
in vec2 v_uv;
out vec4 f_color;
void main() {
    vec3 c = texture(accum, v_uv).rgb;
    if (u_glow > 0.0) c += u_glow * (texture(bloom1, v_uv).rgb * 0.6 + texture(bloom2, v_uv).rgb * 0.9);
    if (u_tonemap == 1) c = 1.0 - exp(-c * 1.2);
    c = clamp(c, 0.0, 1.0);
    float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
    c = clamp(mix(vec3(l), c, u_sat), 0.0, 1.0);
    c = 1.0 - (1.0 - u_bg) * (1.0 - c);
    f_color = vec4(c, 1.0);
}
'''
LINE_VS = '''
#version 330
in vec2 in_pos;
void main() { gl_Position = vec4(in_pos.x, in_pos.y * 2.0 - 1.0, 0.0, 1.0); }
'''
LINE_FS = '''
#version 330
uniform vec4 u_col;
out vec4 f_color;
void main() { f_color = u_col; }
'''


class Renderer:
    def __init__(self, ctx: moderngl.Context, width=1280, height=720):
        self.ctx = ctx
        self.size = (0, 0)
        self.prog_dot = ctx.program(vertex_shader=DOT_VS, fragment_shader=DOT_FS)
        self.prog_blur = ctx.program(vertex_shader=QUAD_VS, fragment_shader=BLUR_FS)
        self.prog_down = ctx.program(vertex_shader=QUAD_VS, fragment_shader=DOWN_FS)
        self.prog_comp = ctx.program(vertex_shader=QUAD_VS, fragment_shader=COMP_FS)
        self.prog_line = ctx.program(vertex_shader=LINE_VS, fragment_shader=LINE_FS)
        quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype='f4').tobytes())
        self.quad = {p: ctx.vertex_array(p, [(quad, '2f', 'in_vert')]) for p in
                     (self.prog_blur, self.prog_down, self.prog_comp)}
        self.vbo = ctx.buffer(reserve=1 << 20, dynamic=True)
        self.vao = ctx.vertex_array(self.prog_dot, [(self.vbo, '2f 1f 3f', 'in_pos', 'in_i', 'in_col')])
        self.line_vbo = ctx.buffer(reserve=1 << 16, dynamic=True)
        self.line_vao = ctx.vertex_array(self.prog_line, [(self.line_vbo, '2f', 'in_pos')])
        self.resize(width, height)

    def resize(self, w, h):
        w, h = max(16, int(w)), max(16, int(h))
        if (w, h) == self.size:
            return
        self.size = (w, h)
        ctx = self.ctx
        for name in ('accum', 'b1a', 'b1b', 'b2a', 'b2b', 'out'):
            obj = getattr(self, name + '_fbo', None)
            if obj is not None:
                obj.release()
                getattr(self, name + '_tex').release()

        def mk(name, sz, dtype='f2'):
            tex = ctx.texture(sz, 4, dtype=dtype)
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex.repeat_x = tex.repeat_y = False
            setattr(self, name + '_tex', tex)
            setattr(self, name + '_fbo', ctx.framebuffer(color_attachments=[tex]))
        mk('accum', (w, h))
        mk('b1a', (max(1, w // 2), max(1, h // 2)))
        mk('b1b', (max(1, w // 2), max(1, h // 2)))
        mk('b2a', (max(1, w // 8), max(1, h // 8)))
        mk('b2b', (max(1, w // 8), max(1, h // 8)))
        mk('out', (w, h), 'f1')

    @property
    def texture(self):
        return self.out_tex

    def render(self, dots):
        """dots: dict from Engine.gather. Leaves the result in self.out_tex."""
        ctx = self.ctx
        w, h = self.size
        r = dots['render']
        n = len(dots['x'])
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.accum_fbo.use()
        ctx.viewport = (0, 0, w, h)
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        if n:
            data = np.empty((n, 6), dtype='f4')
            data[:, 0], data[:, 1], data[:, 2] = dots['x'], dots['y'], dots['i']
            data[:, 3:] = dots['rgb']
            raw = data.tobytes()
            if len(raw) > self.vbo.size:
                self.vbo.orphan(len(raw) * 2)
            self.vbo.write(raw)
            shape = {'square': 0, 'round': 1, 'glow': 2}.get(r['dot_shape'], 0)
            size = float(r['dot_size']) * h / 1080.0 * (3.0 if shape == 2 else 1.0)
            self.prog_dot['u_size'].value = max(1.0, size)
            self.prog_dot['u_shape'].value = shape
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.ONE, moderngl.ONE
            ctx.blend_equation = moderngl.FUNC_ADD
            self.vao.render(moderngl.POINTS, vertices=n)
            ctx.disable(moderngl.BLEND)
        glow = float(r['glow'])
        if glow > 0:
            rad = float(r['glow_radius'])
            self._down(self.accum_tex, self.b1a_fbo)
            self._blur(self.b1a_tex, self.b1b_fbo, (rad / self.b1a_tex.size[0], 0))
            self._blur(self.b1b_tex, self.b1a_fbo, (0, rad / self.b1a_tex.size[1]))
            self._down(self.b1a_tex, self.b2a_fbo, scale=4)
            self._blur(self.b2a_tex, self.b2b_fbo, (rad / self.b2a_tex.size[0], 0))
            self._blur(self.b2b_tex, self.b2a_fbo, (0, rad / self.b2a_tex.size[1]))
        self.out_fbo.use()
        ctx.viewport = (0, 0, w, h)
        self.accum_tex.use(0)
        self.b1a_tex.use(1)
        self.b2a_tex.use(2)
        pc = self.prog_comp
        pc['accum'].value, pc['bloom1'].value, pc['bloom2'].value = 0, 1, 2
        pc['u_glow'].value = glow
        pc['u_tonemap'].value = 1 if r['tonemap'] == 'soft' else 0
        pc['u_sat'].value = float(r['saturation'])
        pc['u_bg'].value = tuple(float(c) for c in r['background'][:3])
        self.quad[pc].render(moderngl.TRIANGLE_STRIP)
        if dots.get('overlay'):
            self._overlay(dots['overlay'], r)
        return self.out_tex

    def _down(self, src, dst_fbo, scale=2):
        dst_fbo.use()
        self.ctx.viewport = (0, 0) + dst_fbo.size
        src.use(0)
        self.prog_down['tex'].value = 0
        self.prog_down['u_texel'].value = (1.0 / src.size[0], 1.0 / src.size[1])
        self.quad[self.prog_down].render(moderngl.TRIANGLE_STRIP)

    def _blur(self, src, dst_fbo, direction):
        dst_fbo.use()
        self.ctx.viewport = (0, 0) + dst_fbo.size
        src.use(0)
        self.prog_blur['tex'].value = 0
        self.prog_blur['u_dir'].value = direction
        self.quad[self.prog_blur].render(moderngl.TRIANGLE_STRIP)

    def _overlay(self, items, r):
        w, h = self.size
        aspect = w / h
        ctx = self.ctx
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        for it in items:
            if it[0] != 'ring':
                continue
            _, zx, zy, zr, col, alpha = it
            th = np.linspace(0, 2 * np.pi, 97)
            xs = zx + np.cos(th) * zr * 2 / aspect
            ys = zy + np.sin(th) * zr
            if r.get('flip_x'):
                xs = -xs
            if r.get('flip_y'):
                ys = 1 - ys
            pts = np.stack([xs, ys], 1).astype('f4')
            self.line_vbo.write(pts.tobytes())
            self.prog_line['u_col'].value = (col[0], col[1], col[2], float(alpha))
            ctx.line_width = 2.0
            self.line_vao.render(moderngl.LINE_STRIP, vertices=len(pts))
        ctx.disable(moderngl.BLEND)

    def read_rgb(self):
        """Returns the output image as (h, w, 3) uint8, top row first."""
        w, h = self.size
        raw = self.out_fbo.read(components=3, alignment=1)
        return np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)[::-1]
