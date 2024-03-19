from manim import *
from manim.opengl import *


class OdinokiyOdinochka(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
        square = Square(color=BLUE, side_length=2)
        dot1 = Dot(color=BLUE).shift(UP).shift(LEFT)
        dot2 = Dot(color=BLUE).shift(DOWN).shift(LEFT)
        cylinder = Cylinder(resolution=(100,100), radius=1, height=2)
        cylinder.rotate(angle=PI/2, axis=LEFT)
        cylinder.set_color(RED)
        cylinder.set_stroke(GREEN)
        all = VGroup(square, dot1, dot2)
        trace1 = TracedPath(dot1.get_end)
        trace2 = TracedPath(dot2.get_end)
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.add(axes)
        self.begin_ambient_camera_rotation(90*DEGREES/3)
        self.play(Create(square))
        self.add(dot1, dot2, trace1, trace2)
        self.play(
            # Rotate(dot1, angle=PI, about_point=ORIGIN, axis=UP),
            # Rotate(dot2, angle=PI, about_point=ORIGIN, axis=UP), 
            Rotate(all, angle=2*PI, axis=UP), run_time=10
        )
        self.play(FadeIn(cylinder))
        self.wait(2)
        self.stop_ambient_camera_rotation()

        self.interactive_embed()


class IntroScene(Scene):
    def construct(self):
        square = Square(color=RED).shift(LEFT * 2)
        circle = Circle(color=BLUE).shift(RIGHT * 2)

        self.play(Write(square), Write(circle))

        # moving objects
        self.play(
            square.animate.shift(UP * 0.5),
            circle.animate.shift(DOWN * 0.5)
        )

        # rotating and filling the square (opacity 80%)
        # scaling and filling the circle (opacity 80%)
        self.play(
            square.animate.rotate(PI / 2).set_fill(RED, 0.8),
            circle.animate.scale(2).set_fill(BLUE, 0.8),
        )

        # this is new!
        self.interactive_embed()
