from manim import *
from manim.opengl import *
from manim.typing import Point3D


class Scene1(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
        self.set_camera_orientation(phi=75 * DEGREES)

        intro_text_1 = Text(
            '''
                Сейчас тут будет цилиндрик
                на него можно посмотреть
                и порадоваться ;)
            '''
        )
        self.play(Write(intro_text_1.rotate(PI/2, axis=RIGHT).shift(4*UP)))

        square = Square(color=BLUE, side_length=2)

        dot1 = Dot(color=BLUE)
        dot2 = Dot(color=BLUE)

        all = VGroup(
            square.shift(LEFT),
            dot1.shift(UP).shift(2*LEFT),
            dot2.shift(DOWN).shift(2*LEFT)
        )

        cylinder = Cylinder(
            resolution=(10,50), radius=2, height=2, stroke_width=np.array([0.5])
        )

        self.begin_ambient_camera_rotation(90*DEGREES/3)

        self.play(Create(axes))

        self.play(Create(all))
        self.add(TracedPath(dot1.get_center), TracedPath(dot2.get_center))

        self.play(
            Rotate(all, angle=2*PI, axis=UP, about_point=(0,0,0)), run_time=10
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


class Scene2(ThreeDScene):
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
        surface_plane.set_fill_by_value(axes=axes, colorscale=[(RED, -0.5), (YELLOW, 0), (GREEN, 0.5)], axis=2)
        surface_plane.shift(OUT*0.5)

        self.begin_ambient_camera_rotation(90*DEGREES/3)
        self.play(Create(cone), run_time=2)
        self.play(Create(surface_plane), run_time=5)
        # self.wait(2)
        # self.stop_ambient_camera_rotation()

        self.interactive_embed()


class Scene3(ThreeDScene):
    def construct(self):
        resolution_fa = 8
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        axes = ThreeDAxes()

        def param_surface(u, v) -> float:
            x = u
            y = v
            z = np.sin(x) * np.cos(y)
            return z

        dot1 = Dot3D(axes=axes, point=[0, 2, param_surface(0, 2)], color=BLUE)
        dot2 = Dot3D(axes=axes, point=[2, 0, param_surface(2, 0)], color=BLUE)
        dot3 = Dot3D(axes=axes, point=[0, 0, param_surface(0, 0)], color=BLUE)
        dot4 = Dot3D(axes=axes, point=[2, 2, param_surface(2, 2)], color=BLUE)

        all = VGroup(
            dot1,
            dot2,
            dot3,
            dot4
        )

        surface_plane = Surface(
            lambda u, v: axes.c2p(u, v, param_surface(u, v)),
            resolution=(resolution_fa, resolution_fa),
            v_range=[0, 2],
            u_range=[0, 2],
            # v_range=[-2, 2],
            # u_range=[-2, 2],
            )
        surface_plane.set_style(fill_opacity=1)
        surface_plane.set_fill_by_value(axes=axes, colorscale=[(RED, -0.5), (YELLOW, 0), (GREEN, 0.5)], axis=2)
        self.add(axes)

        self.begin_ambient_camera_rotation(90*DEGREES/3)

        self.play(Create(surface_plane), run_time=3)
        self.play(Create(all))

        self.interactive_embed()


class Scene4(ThreeDScene):
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

        self.play(Rotate(cylinder, angle=0.5*PI, axis=LEFT, about_point=(0,0,0)), run_time=0)
        sphere.shift(-2*OUT)
        sphere.shift(2*UP)
        self.set_camera_orientation(phi=0, theta=0)
    
        self.play(Transform(cylinder, Rectangle(height=2, width=1.4, color=GREEN).shift(UP)))
        self.remove(sphere)
        self.play(FadeIn(Circle(radius=1, color=ORANGE).shift(2*UP)))

        self.interactive_embed()
