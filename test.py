from re import L
from manim import *
from manim.opengl import *
from manim.typing import Point3D


# Вступление
class Frame1(Scene):
    def construct(self):
        intro_text = 'Маним - python библиотека'
        base_text = (
            '<span text-align="center">'
            'С помощью этой библиотеки '
            'можно визуализировать '
            'сложные для объяснения '
            'математические понятия '
            '</span>'
        )
        intro = MarkupText(intro_text, gradient=(
            BLUE, GREEN, YELLOW, ORANGE, RED))
        base = MarkupText(base_text, should_center=True, font_size=100, gradient=(
            BLUE, GREEN, YELLOW, ORANGE, RED)).scale(0.4)

        ds_m = MathTex(r"\mathbb{M}", fill_color=WHITE).scale(7)
        ds_m.shift(2.25 * LEFT + 1.5 * UP)
        circle = Circle(color=GREEN, fill_opacity=1).shift(LEFT)
        square = Square(color=BLUE, fill_opacity=1).shift(UP)
        triangle = Triangle(color=RED, fill_opacity=1).shift(RIGHT)
        logo = VGroup(triangle, square, circle, ds_m)
        logo.move_to(ORIGIN)

        self.play(Write(intro))
        self.wait(1)
        self.play(Transform(intro, logo))
        self.wait(1)
        self.play(Transform(intro, base))
        self.wait(2)


class Frame2(Scene):
    def construct(self):
        text = ("В качестве примера продемонстрируем нахождение объема тела вращения с помощью интеграла")
        base = MarkupText(text, should_center=True, font_size=100, gradient=(
            BLUE, GREEN)).scale(0.3)
        tex = MathTex(r"V = \int_{n}^{m} f(x) dx", font_size=50)

        self.play(Write(base.shift(UP)))
        self.wait(1)
        self.play(Write(tex.shift(DOWN)))
        self.wait(1)


class Frame3(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
        self.set_camera_orientation(phi=75 * DEGREES)

        intro_text_1 = Text(
            '''
                Объем тела вращения
            '''
        )
        self.play(Write(intro_text_1.rotate(PI/2, axis=RIGHT).shift(4*UP)))
        self.wait(1)
        self.play(FadeOut(intro_text_1))

        square = Square(color=BLUE, side_length=2)

        dot1 = Dot(color=BLUE)
        dot2 = Dot(color=BLUE)

        all = VGroup(
            square.shift(LEFT),
            dot1.shift(UP).shift(2*LEFT),
            dot2.shift(DOWN).shift(2*LEFT)
        )

        cylinder = Cylinder(
            resolution=(10, 50), radius=2, height=2, stroke_width=np.array([0.5])
        )

        self.begin_ambient_camera_rotation(90*DEGREES/3)

        self.play(Create(axes))

        self.play(Create(all))
        self.add(TracedPath(dot1.get_center), TracedPath(dot2.get_center))

        self.play(
            Rotate(all, angle=2*PI, axis=UP, about_point=(0, 0, 0)), run_time=10
        )

        self.play(FadeIn(
            cylinder.rotate(angle=PI/2, axis=LEFT)
            .set_opacity(0.4)
            .set_color(YELLOW)
            .set_stroke(GREEN)
        ))
        # self.wait(2)
        # self.stop_ambient_camera_rotation()

        self.interactive_embed()


class Frame4(Scene):
    def construct(self):
        text = ("Нахождение объема под графиком")
        base = MarkupText(text, should_center=True, font_size=100, gradient=(
            GREEN, YELLOW)).scale(0.3)
        tex = MathTex(r"V = \iint_{D} f(x, y) dx dy", font_size=50)

        self.play(Write(base.shift(UP)))
        self.wait(1)
        self.play(Write(tex.shift(DOWN)))
        self.wait(1)


class Frame5(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
        cone = Cone(base_radius=0.5, height=2)
        cone.set_color(GREEN)
        cone.set_opacity(0.4)
        cone.shift(2*OUT)
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.add(axes)

        resolution_fa = 20

        def param_surface(u, v):
            x = u
            z = x
            return z
        surface_plane = Surface(
            lambda u, v: axes.c2p(u, v, param_surface(u, v)),
            resolution=(resolution_fa, resolution_fa),
            v_range=[-1, 1],
            u_range=[-1, 1],
        )
        surface_plane.set_style(fill_opacity=0.4)
        surface_plane.set_fill_by_value(
            axes=axes, colorscale=[(RED, -0.5), (YELLOW, 0), (GREEN, 0.5)], axis=2)
        surface_plane.shift(OUT*0.5)

        self.begin_ambient_camera_rotation(90*DEGREES/3)
        self.play(Create(cone), run_time=2)
        self.play(Create(surface_plane), run_time=5)
        self.wait(2)
        self.stop_ambient_camera_rotation()

        self.camera.set_focal_distance(1000)
        self.set_camera_orientation(phi=PI/2, theta=PI/2)
        self.wait(2)

        self.interactive_embed()


class Frame6(Scene):
    def construct(self):
        text = ("Нахождение объема под графиком")
        base = MarkupText(text, should_center=True, font_size=100, gradient=(
            YELLOW, ORANGE)).scale(0.3)
        tex = MathTex(r"V = \iint_{D} f(x, y) dx dy", font_size=50)

        self.play(Write(base.shift(UP)))
        self.wait(1)
        self.play(Write(tex.shift(DOWN)))
        self.wait(1)


class Frame7(ThreeDScene):
    def construct(self):
        resolution_fa = 8
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        axes = ThreeDAxes()

        def param_surface(u, v) -> float:
            x = u
            y = v
            z = np.sin(x) * np.cos(y)
            return z

        dot1 = Dot3D(point=axes.c2p(-2, 2, param_surface(-2, 2)), color=BLUE)
        dot2 = Dot3D(point=axes.c2p(2, -2, param_surface(2, -2)), color=BLUE)
        dot3 = Dot3D(point=axes.c2p(-2, -2, param_surface(-2, -2)), color=BLUE)
        dot4 = Dot3D(point=axes.c2p(2, 2, param_surface(2, 2)), color=BLUE)

        all = VGroup(
            dot1,
            dot2,
            dot3,
            dot4
        )

        dot1.generate_target()
        dot1.target.move_to(Dot3D(point=axes.c2p(-2, 2, 0)))
        dot2.generate_target()
        dot2.target.move_to(Dot3D(point=axes.c2p(2, -2, 0)))
        dot3.generate_target()
        dot3.target.move_to(Dot3D(point=axes.c2p(-2, -2, 0)))
        dot4.generate_target()
        dot4.target.move_to(Dot3D(point=axes.c2p(2, 2, 0)))

        surface_plane = Surface(
            lambda u, v: axes.c2p(u, v, param_surface(u, v)),
            resolution=(resolution_fa, resolution_fa),
            v_range=[-2, 2],
            u_range=[-2, 2],
        )
        surface_plane.set_style(fill_opacity=1)
        surface_plane.set_fill_by_value(
            axes=axes, colorscale=[(RED, -0.5), (YELLOW, 0), (GREEN, 0.5)], axis=2)
        self.add(axes)

        self.begin_ambient_camera_rotation(90*DEGREES/3)

        self.play(Create(surface_plane.shift(2*OUT)), run_time=3)
        self.play(Create(all.shift(2*OUT)))
        self.add(
            TracedPath(dot1.get_center),
            TracedPath(dot2.get_center),
            TracedPath(dot3.get_center),
            TracedPath(dot4.get_center)
        )
        self.play(MoveToTarget(dot1))
        self.play(MoveToTarget(dot2))
        self.play(MoveToTarget(dot3))
        self.play(MoveToTarget(dot4))

        self.stop_ambient_camera_rotation()

        self.camera.set_focal_distance(1000)
        self.set_camera_orientation(phi=PI/2, theta=PI/2)
        self.wait(2)

        self.interactive_embed()


class Frame8(Scene):
    def construct(self):
        text = ("Нахождение объема ограниченного несколькими фигурами")
        base = MarkupText(text, should_center=True, font_size=100, gradient=(
            ORANGE, RED)).scale(0.3)
        tex = MathTex(r"V = \iiint_{D} f(x,y,z) dx dy dz", font_size=50)

        self.play(Write(base.shift(UP)))
        self.wait(1)
        self.play(Write(tex.shift(DOWN)))
        self.wait(1)


class Frame9(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
        sphere = Sphere(radius=1)
        sphere.set_color(ORANGE)
        sphere.set_opacity(0.5)
        sphere.shift(2*OUT)

        cylinder = Cylinder(radius=0.7, height=2, stroke_width=np.array([0.5]))
        cylinder.set_color(GREEN)
        cylinder.set_opacity(0.4)
        cylinder.shift(OUT)

        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.add(axes)
        self.begin_ambient_camera_rotation(90*DEGREES/3)
        all = VGroup(
            cylinder,
            sphere
        )

        self.play(Create(cylinder), run_time=2)
        self.play(Create(sphere), run_time=2)

        self.wait(2)
        self.stop_ambient_camera_rotation()

        self.play(Rotate(cylinder, angle=0.5*PI, axis=LEFT,
                  about_point=(0, 0, 0)), run_time=0)
        sphere.shift(-2*OUT)
        sphere.shift(2*UP)
        self.set_camera_orientation(phi=0, theta=PI/2*3)

        self.play(Transform(cylinder, Rectangle(
            height=2, width=1.4, color=GREEN).shift(UP)))
        self.remove(sphere)
        self.play(FadeIn(Circle(radius=1, color=ORANGE).shift(2*UP)))

        self.wait(2)

        self.interactive_embed()
