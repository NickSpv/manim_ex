from manim import *
from manim.opengl import *
from math import floor
import random
from lookup import LookupTable
import itertools
np.random.seed(0)
import time


class OdinokiyOdinochka(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
        square = Square(color=BLUE, side_length=2)
        square.shift(LEFT)
        dot1 = Dot(color=BLUE).shift(UP).shift(2*LEFT)
        dot2 = Dot(color=BLUE).shift(DOWN).shift(2*LEFT)
        cylinder = Cylinder(
            resolution=(100,100), radius=1, height=2, stroke_width=np.array([0.5])
        )
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
        self.wait(5)
        self.stop_ambient_camera_rotation()

        self.interactive_embed()


class OdinokiyOdinochkaIshshoOdin(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
        square = Square(color=BLUE, side_length=2)
        dot1 = Dot(color=BLUE).shift(UP).shift(LEFT)
        dot2 = Dot(color=BLUE).shift(DOWN).shift(LEFT)
        all = VGroup(square, dot1, dot2)
        trace1 = TracedPath(dot1.get_start)
        trace2 = TracedPath(dot2.get_start)
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
        self.wait(2)
        self.stop_ambient_camera_rotation()


class Part0(Scene):
    def construct(self):
        intro_text_1 = Text('Это видео демонстрирует', should_center=True)
        intro_text_2 = Text('возможности python библиотеки manim', should_center=True)
        intro_text_1.shift(UP)
        self.play(Write(intro_text_1))
        self.play(Write(intro_text_2))
        self.wait()


class Part1(Scene):
    def construct(self):
        intro_text_1 = Text('Визуализация latex формул', should_center=True)
        intro_text_1.shift(UP*3)
        self.play(Write(intro_text_1))
        text=MathTex(
            "\\frac{d}{dx}f(x)g(x)=","f(x)\\frac{d}{dx}g(x)","+",
            "g(x)\\frac{d}{dx}f(x)"
        )
        text.shift(DOWN)
        self.play(Write(text))
        framebox1 = SurroundingRectangle(text[1], buff = .1)
        framebox2 = SurroundingRectangle(text[3], buff = .1)
        self.play(
            Create(framebox1),
        )
        self.wait()
        self.play(
            ReplacementTransform(framebox1,framebox2),
        )
        self.wait()
        self.wait()


class Part2(Scene):
    def get_rectangle_corners(self, bottom_left, top_right):
        return [
            (top_right[0], top_right[1]),
            (bottom_left[0], top_right[1]),
            (bottom_left[0], bottom_left[1]),
            (top_right[0], bottom_left[1]),
        ]

    def construct(self):
        intro_text_1 = Text('Визуализация различных графиков', should_center=True)
        intro_text_2 = Text('с динамическим выделением чего-либо', should_center=True)
        intro_text_1.shift(UP)
        self.play(Write(intro_text_1))
        self.play(Write(intro_text_2))
        self.wait()
        self.play(FadeOut(intro_text_1), run_time = 0.5)
        self.play(FadeOut(intro_text_2), run_time = 0.5)
        ax = Axes(
            x_range=[0, 10],
            y_range=[0, 10],
            x_length=6,
            y_length=6,
            axis_config={"include_tip": False},
        )

        t = ValueTracker(5)
        k = 25

        graph = ax.plot(
            lambda x: k / x,
            color=YELLOW_D,
            x_range=[k / 10, 10.0, 0.01],
            use_smoothing=False,
        )

        def get_rectangle():
            polygon = Polygon(
                *[
                    ax.c2p(*i)
                    for i in self.get_rectangle_corners(
                        (0, 0), (t.get_value(), k / t.get_value())
                    )
                ]
            )
            polygon.stroke_width = 1
            polygon.set_fill(BLUE, opacity=0.5)
            polygon.set_stroke(YELLOW_B)
            return polygon

        polygon = always_redraw(get_rectangle)

        dot = Dot()
        dot.add_updater(lambda x: x.move_to(ax.c2p(t.get_value(), k / t.get_value())))
        dot.set_z_index(10)

        self.add(ax, graph, dot)
        self.play(Create(polygon))
        self.play(t.animate.set_value(10))
        self.play(t.animate.set_value(k / 10))
        self.play(t.animate.set_value(5))


class Part3(ZoomedScene):
    def __init__(self, **kwargs):
        ZoomedScene.__init__(
            self,
            zoom_factor=0.3,
            zoomed_display_height=1,
            zoomed_display_width=6,
            image_frame_stroke_width=20,
            zoomed_camera_config={
                "default_frame_stroke_width": 3,
                },
            **kwargs
        )

    def construct(self):
        intro_text_1 = Text('Приближение части какой-либо картинки', should_center=True)
        self.play(Write(intro_text_1))
        self.wait()
        self.play(FadeOut(intro_text_1), run_time = 0.5)
        dot = Dot().shift(UL * 2)
        image = ImageMobject(np.uint8([[0, 100, 30, 200],
                                       [255, 0, 5, 33]]))
        image.height = 7
        frame_text = Text("Часть картинки", color=PURPLE, font_size=67)
        zoomed_camera_text = Text("Увеличение", color=RED, font_size=67)

        self.add(image, dot)
        zoomed_camera = self.zoomed_camera
        zoomed_display = self.zoomed_display
        frame = zoomed_camera.frame
        zoomed_display_frame = zoomed_display.display_frame

        frame.move_to(dot)
        frame.set_color(PURPLE)
        zoomed_display_frame.set_color(RED)
        zoomed_display.shift(DOWN)

        zd_rect = BackgroundRectangle(zoomed_display, fill_opacity=0, buff=MED_SMALL_BUFF)
        self.add_foreground_mobject(zd_rect)

        unfold_camera = UpdateFromFunc(zd_rect, lambda rect: rect.replace(zoomed_display))

        frame_text.next_to(frame, DOWN)

        self.play(Create(frame), FadeIn(frame_text, shift=UP))
        self.activate_zooming()

        self.play(self.get_zoomed_display_pop_out_animation(), unfold_camera)
        zoomed_camera_text.next_to(zoomed_display_frame, DOWN)
        self.play(FadeIn(zoomed_camera_text, shift=UP))
        # Scale in        x   y  z
        scale_factor = [0.5, 1.5, 0]
        self.play(
            frame.animate.scale(scale_factor),
            zoomed_display.animate.scale(scale_factor),
            FadeOut(zoomed_camera_text),
            FadeOut(frame_text)
        )
        self.wait()
        self.play(ScaleInPlace(zoomed_display, 2))
        self.wait()
        self.play(frame.animate.shift(2.5 * DOWN))
        self.wait()
        self.play(self.get_zoomed_display_pop_out_animation(), unfold_camera, rate_func=lambda t: smooth(1 - t))
        self.play(Uncreate(zoomed_display_frame), FadeOut(frame))
        self.wait()


class Part4(Scene):
    def construct(self):
        intro_text_1 = Text('Использование latex', should_center=True)
        self.play(Write(intro_text_1))
        self.wait()
        self.play(FadeOut(intro_text_1), run_time = 0.5)
        title = Tex(r"This is some \LaTeX")
        basel = MathTex(r"\sum_{n=1}^\infty \frac{1}{n^2} = \frac{\pi^2}{6}")
        VGroup(title, basel).arrange(DOWN)
        self.play(
            Write(title),
            FadeIn(basel, shift=DOWN),
        )
        self.wait()

        transform_title = Tex("That was a transform")
        transform_title.to_corner(UP + LEFT)
        self.play(
            Transform(title, transform_title),
            LaggedStart(*[FadeOut(obj, shift=DOWN) for obj in basel]),
        )
        self.wait()

        grid = NumberPlane()
        grid_title = Tex("This is a grid", font_size=72)
        grid_title.move_to(transform_title)

        self.add(grid, grid_title)  # Make sure title is on top of grid
        self.play(
            FadeOut(title),
            FadeIn(grid_title, shift=UP),
            Create(grid, run_time=3, lag_ratio=0.1),
        )
        self.wait()

        grid_transform_title = Tex(
            r"That was a non-linear function \\ applied to the grid"
        )
        grid_transform_title.move_to(grid_title, UL)
        grid.prepare_for_nonlinear_transform()
        self.play(
            grid.animate.apply_function(
                lambda p: p
                          + np.array(
                    [
                        np.sin(p[1]),
                        np.sin(p[0]),
                        0,
                    ]
                )
            ),
            run_time=3,
        )
        self.wait()
        self.play(Transform(grid_title, grid_transform_title))
        self.wait()


class Part5(Scene):
    def construct(self):
        intro_text_1 = Text('Тригонометрия', should_center=True)
        self.play(Write(intro_text_1))
        self.wait()
        self.play(FadeOut(intro_text_1), run_time = 0.5)
        self.show_axis()
        self.show_circle()
        self.move_dot_and_draw_curve()
        self.wait()

    def show_axis(self):
        x_start = np.array([-6,0,0])
        x_end = np.array([6,0,0])

        y_start = np.array([-4,-2,0])
        y_end = np.array([-4,2,0])

        x_axis = Line(x_start, x_end)
        y_axis = Line(y_start, y_end)

        self.add(x_axis, y_axis)
        self.add_x_labels()

        self.origin_point = np.array([-4,0,0])
        self.curve_start = np.array([-3,0,0])

    def add_x_labels(self):
        x_labels = [
            MathTex("\pi"), MathTex("2 \pi"),
            MathTex("3 \pi"), MathTex("4 \pi"),
        ]

        for i in range(len(x_labels)):
            x_labels[i].next_to(np.array([-1 + 2*i, 0, 0]), DOWN)
            self.add(x_labels[i])

    def show_circle(self):
        circle = Circle(radius=1)
        circle.move_to(self.origin_point)
        self.add(circle)
        self.circle = circle

    def move_dot_and_draw_curve(self):
        orbit = self.circle
        origin_point = self.origin_point

        dot = Dot(radius=0.08, color=YELLOW)
        dot.move_to(orbit.point_from_proportion(0))
        self.t_offset = 0
        rate = 0.25

        def go_around_circle(mob, dt):
            self.t_offset += (dt * rate)
            # print(self.t_offset)
            mob.move_to(orbit.point_from_proportion(self.t_offset % 1))

        def get_line_to_circle():
            return Line(origin_point, dot.get_center(), color=BLUE)

        def get_line_to_curve():
            x = self.curve_start[0] + self.t_offset * 4
            y = dot.get_center()[1]
            return Line(dot.get_center(), np.array([x,y,0]), color=YELLOW_A, stroke_width=2 )


        self.curve = VGroup()
        self.curve.add(Line(self.curve_start,self.curve_start))
        def get_curve():
            last_line = self.curve[-1]
            x = self.curve_start[0] + self.t_offset * 4
            y = dot.get_center()[1]
            new_line = Line(last_line.get_end(),np.array([x,y,0]), color=YELLOW_D)
            self.curve.add(new_line)

            return self.curve

        dot.add_updater(go_around_circle)

        origin_to_circle_line = always_redraw(get_line_to_circle)
        dot_to_curve_line = always_redraw(get_line_to_curve)
        sine_curve_line = always_redraw(get_curve)

        self.add(dot)
        self.add(orbit, origin_to_circle_line, dot_to_curve_line, sine_curve_line)
        self.wait(8.5)

        dot.remove_updater(go_around_circle)


class Part6(ThreeDScene):
    def construct(self):
        intro_text_1 = Text('Трехмерные графики', should_center=True)
        self.play(Write(intro_text_1))
        self.wait()
        self.play(FadeOut(intro_text_1), run_time = 0.5)
        resolution_fa = 24
        axes = ThreeDAxes()
        def param_gauss(u, v):
            x = u
            y = v
            sigma, mu = 0.4, [0.0, 0.0]
            d = np.linalg.norm(np.array([x - mu[0], y - mu[1]]))
            z = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))
            return np.array([x, y, z])
        gauss_plane = Surface(
            param_gauss,
            resolution=(resolution_fa, resolution_fa),
            v_range=[-2, +2],
            u_range=[-2, +2]
        )
        gauss_plane.scale(2, about_point=ORIGIN)
        gauss_plane.set_style(fill_opacity=1,stroke_color=GREEN)
        gauss_plane.set_fill_by_checkerboard(ORANGE, BLUE, opacity=0.5)
        axes = ThreeDAxes()
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.add(axes,gauss_plane)
        self.begin_3dillusion_camera_rotation(rate=2)
        self.wait(5)
        self.stop_3dillusion_camera_rotation()


INSIDE_COLOR = PURE_GREEN
OUTSIDE_COLOR = BLUE
CONTOUR_COLOR = YELLOW
class MarchingSquare:
    NORTH = (0, 1)
    EAST = (1, 2)
    SOUTH = (2, 3)
    WEST = (3, 0)

    lookup_table = {
    1: [(SOUTH, WEST),],
    2: [(EAST, SOUTH),],
    3: [(EAST, WEST),],
    4: [(NORTH, EAST),],
    5: [(WEST, NORTH), (EAST, SOUTH)],
    6: [(NORTH, SOUTH),],
    7: [(WEST, NORTH)],
    8: [(WEST, NORTH)],
    9: [(NORTH, SOUTH)],
    10: [(NORTH, EAST), (SOUTH, WEST)],
    11: [(NORTH, EAST)],
    12: [(EAST, WEST)],
    13: [(EAST, SOUTH)],
    14: [(SOUTH, WEST)],
    }

    lookup_table_polygons = {
    1: [SOUTH, 3,  WEST,],
    2: [EAST, 2, SOUTH,],
    3: [EAST, 2, 3, WEST],
    4: [NORTH, 1, EAST],
    5: [WEST, NORTH, 1, EAST, SOUTH, 3],
    6: [NORTH, 1, 2, SOUTH],
    7: [WEST, NORTH, 1, 2, 3],
    8: [WEST, 0, NORTH],
    9: [NORTH, 0, 3, SOUTH],
    10: [NORTH, EAST, 2, SOUTH, WEST, 0] ,
    11: [NORTH, EAST, 2, 3, 0],
    12: [EAST, 1, 0, WEST],
    13: [EAST, SOUTH, 3, 0, 1],
    14: [SOUTH, WEST, 0, 1, 2],
    15: [0, 1, 2, 3]
    }
    def __init__(self, ul, ur, dr, dl, function_map):
        self.corners = [ul, ur, dr, dl]
        self.function_map = function_map
    
    def get_corners(self):
        return self.corners

    def get_lines_for_case(self, case, implicit_function, 
        value=1, width=2, gradient=False, color=CONTOUR_COLOR):
        lines = VGroup()
        edges_to_connect = MarchingSquare.lookup_table[case]
        for edge1, edge2 in edges_to_connect:
            start = self.get_point_on_edge(edge1, implicit_function, value=value)
            end = self.get_point_on_edge(edge2, implicit_function, value=value)
            line = Line(start, end, color=color).set_stroke(width=width)
            if gradient:
                line.set_stroke(color=position_to_color(start))
            lines.add(line)

        return lines

    def get_polygon_for_case(self, case, implicit_function, value=1, gradient=False):
        polygon_identifier = MarchingSquare.lookup_table_polygons[case]
        return self.get_polygon(polygon_identifier, implicit_function, value=value, gradient=gradient)

    def get_polygon(self, identifier, implicit_function, value=1, gradient=False):
        polygon_points = []
        for point in identifier:
            if isinstance(point, int):
                polygon_points.append(self.corners[point])
            else:
                edge = point
                point_on_edge = self.get_point_on_edge(edge, implicit_function, value=value)
                polygon_points.append(point_on_edge)
        polygon = Polygon(*polygon_points)
        if not gradient:
            polygon.set_stroke(color=CONTOUR_COLOR, width=1)
            polygon.set_fill(color=CONTOUR_COLOR, opacity=1)
        else:
            color = position_to_color(polygon_points[0])
            polygon.set_stroke(color=color, width=1)
            polygon.set_fill(color=color, opacity=1)
        return polygon

    def get_point_on_edge(self, edge, implicit_function, value=1):
        u, v = edge
        u_pos, v_pos = self.corners[u], self.corners[v]
        value_u_pos = get_func_val_from_map(self.function_map, (u_pos[0], u_pos[1]), implicit_function)
        value_v_pos = get_func_val_from_map(self.function_map, (v_pos[0], v_pos[1]), implicit_function)

        return self.lerp(u_pos, v_pos, value_u_pos, value_v_pos, value=value)

    def lerp(self, u_pos, v_pos, value_u_pos, value_v_pos, value=1):
        if value_u_pos == value_v_pos:
            return (u_pos + v_pos) / 2
        if value_u_pos < value_v_pos:
            t = (value - value_u_pos) / (value_v_pos - value_u_pos)
            return (1 - t) * u_pos + t * v_pos
        else:
            t = (value - value_v_pos) / (value_u_pos - value_v_pos)
            return (1 - t) * v_pos + t * u_pos

class MarchingSquaresUtils(Scene):
    # Utilities class for performing marching squares
    def construct(self):
        pass

    def set_sample_space(self, x_range=(-7.5, 7.5), y_range=(-4, 4), x_step=0.5, y_step=0.5,):
        """
        @param: x_range - (x_min, x_max) by default the min and max of the screen
        @param: y_range - (y_min, y_max) by default the min and max of the screen
        @param: x_step  - step between samples in x direction
        @param: y_step  - step between samples in y direction
        """
        self.sample_space = []
        self.x_step = x_step
        self.y_step = y_step
        self.x_min = x_range[0]
        self.x_max = x_range[1]
        self.y_min = y_range[0]
        self.y_max = y_range[1]
        for x in np.arange(*self.get_iterable_range(x_range, x_step)):
            for y in np.arange(*self.get_iterable_range(y_range, y_step)):
                self.sample_space.append(np.array([x, y, 0.]))
        self.sample_space = np.array(self.sample_space)
    
    def get_sample_space(self):
        """
        Returns current sample space
        """
        return self.sample_space

    def get_iterable_range(self, standard_range, step):
        """
        @param: standard_range: a python range tuple (start, end) with start inclusive, end excluse
        @step: step to sample range
        @return: tuple (start, end, step) where end will include current excluded value 
        """
        return (standard_range[0], standard_range[1] + step, step)

    def get_implicit_function_samples(self, func):
        """
        @param: func - the implicit function to sample
        @return: samples of implicit function across sample space
        """
        self.function_map = {}
        for sample_point in self.sample_space:
            result = func(sample_point)
            x, y, _ = sample_point
            self.function_map[(x, y)] = result 

        return self.function_map

    def get_dots_based_on_condition(self, condition, radius=0.05):
        """
        @param: condition - function which defines boundary for when point is on surface/contour
        @param: radius - radius of the dot
        @return: VGroup of dots
        """
        self.dot_map = {}
        dots = VGroup()
        for key in self.function_map:
            position = np.array([key[0], key[1], 0])
            if condition(self.function_map[key]):
                dot = Dot(radius=radius, color=INSIDE_COLOR).move_to(position)
            else:
                dot = Dot(radius=radius, color=OUTSIDE_COLOR).move_to(position)
            self.dot_map[key] = dot
            dots.add(dot)
        return dots

    def get_values_of_implicit_f(self, scale=0.3):
        values = VGroup()
        self.decimal_map = {}
        for key in self.function_map:
            value = DecimalNumber(number=self.function_map[key], num_decimal_places=2).scale(scale)
            value.next_to(self.dot_map[key], DR, buff=0)
            self.decimal_map[key] = value
            values.add(value)
        return values


    def update_dots(self, condition):
        for key in self.function_map:
            if condition(self.function_map[key]):
                self.dot_map[key].set_color(INSIDE_COLOR)
            else:
                self.dot_map[key].set_color(OUTSIDE_COLOR)

    def update_values(self):
        for key in self.function_map:
            self.decimal_map[key].set_value(self.function_map[key])

    def march_squares(self, condition, implicit_function, value=1, line_width=2, gradient=False, fill=False, color=CONTOUR_COLOR):
        contour = VGroup()
        count = 0
        for key in self.function_map:
            if key[1] >= self.y_max:
                continue
            if key[0] >= self.x_max:
                continue

            # if count > 100:
            #     break
            marching_square = self.get_marching_square(key)
            square_corners = marching_square.get_corners()
            case = self.get_case(square_corners, condition, implicit_function)
            # case_integer = Integer(case).scale(0.4)
            # case_integer.move_to(np.mean(square_corners, axis=0))
            # self.add(case_integer)
            lines = self.process_case(case, marching_square, implicit_function, 
                value=value, width=line_width, gradient=gradient, fill=fill, color=color)
            if lines:
                contour.add(lines)
            # print(case)
            count += 1
        return contour

    def get_marching_square(self, key):
        ul_pos = np.array([key[0], key[1] + self.y_step, 0])
        ur_pos = np.array([key[0] + self.x_step, key[1] + self.y_step, 0])
        dr_pos = np.array([key[0] + self.x_step, key[1], 0])
        dl_pos = np.array([key[0], key[1], 0])
        return MarchingSquare(ul_pos, ur_pos, dr_pos, dl_pos, self.function_map)

    def get_case(self, square_corners, condition, implicit_function):
        bin_string = ''
        for corner in square_corners:
            value = get_func_val_from_map(self.function_map, (corner[0], corner[1]), implicit_function)
            # print(value)
            if condition(value):
                bin_string += '1'
            else:
                bin_string += '0'
        # print(bin_string)
        return int(bin_string, 2)

    def process_case(self, case, marching_square, implicit_function, 
        value=1, width=2, gradient=False, fill=False, color=CONTOUR_COLOR):
        """
        Draws lines based on the case of marching cubes
        """
        if fill:
            if case == 0:
                return
            polygons = marching_square.get_polygon_for_case(
                case, implicit_function,
                value=value, gradient=gradient
            )
            return polygons
        if case == 0 or case == 15:
            return
        lines = marching_square.get_lines_for_case(
            case, implicit_function, 
            value=value, width=width, gradient=gradient, color=color
        )
        return lines

class MarchingCube:
    # TODO: Test this class!!!!!
    # Maps edge number to (u, v) where u and v are vertex numbers
    EDGES = {
    0: (0, 1),
    1: (1, 2),
    2: (2, 3),
    3: (3, 0),
    4: (4, 5),
    5: (5, 6),
    6: (6, 7),
    7: (7, 4),
    8: (0, 4),
    9: (1, 5),
    10: (2, 6),
    11: (3, 7),
    }

    TRIANGLE_TABLE = LookupTable.TABLE

    def __init__(self, corners, function_map, implicit_function):
        # corners - [v0, v1, v2, v3, v4, v5, v6, v7]
        # each v_i is a point in 3D space [x, y, z]
        self.corners = corners
        self.function_map = function_map
        self.implicit_function = implicit_function
    
    def get_corners(self):
        return self.corners

    def get_side_length(self):
        return np.linalg.norm(self.corners[1] - self.corners[0])

    def get_cube_index(self, condition):
        # Calculates the appropriate key to lookup in the TRIANGLE_TABLE
        index = 0
        for i, point in enumerate(self.corners):
            value = get_func_val_from_map_3d(self.function_map, (point[0], point[1], point[2]), self.implicit_function)
            if condition(value):
                index |= 2 ** i
        return index

    def get_triangles_for_case(self, cube_index):
        triangles = VGroup()
        triangles_to_add = MarchingCube.TRIANGLE_TABLE[cube_index]
        index = 0
        while triangles_to_add[index] != -1:
            edge1, edge2, edge3 = triangles_to_add[index], triangles_to_add[index + 1], triangles_to_add[index + 2]
            p1 = self.get_point_on_edge(edge1)
            p2 = self.get_point_on_edge(edge2)
            p3 = self.get_point_on_edge(edge3)
            polygon = Polygon(p1, p2, p3, color=CONTOUR_COLOR).set_stroke(width=2)
            # polygon.set_fill(color=color, opacity=1)
            triangles.add(polygon)
            index += 3

        return triangles
    
    def get_point_on_edge(self, edge):
        u, v = MarchingCube.EDGES[edge]
        u_pos, v_pos = self.corners[u], self.corners[v]
        value_u_pos = get_func_val_from_map(self.function_map, (u_pos[0], u_pos[1], u_pos[2]), self.implicit_function)
        value_v_pos = get_func_val_from_map(self.function_map, (v_pos[0], v_pos[1], v_pos[2]), self.implicit_function)

        return self.lerp(u_pos, v_pos, value_u_pos, value_v_pos)

    def lerp(self, u_pos, v_pos, value_u_pos, value_v_pos):
        if value_u_pos == value_v_pos:
            return (u_pos + v_pos) / 2
        if value_u_pos < value_v_pos:
            t = (1 - value_u_pos) / (value_v_pos - value_u_pos)
            return (1 - t) * u_pos + t * v_pos
        else:
            t = (1 - value_v_pos) / (value_u_pos - value_v_pos)
            return (1 - t) * v_pos + t * u_pos



class MarchingCubesUtils(ThreeDScene):
    def construct(self):
        pass

    def set_sample_space(self, x_range=(-7.5, 7.5), y_range=(-4, 4), z_range=(-4, 4), x_step=0.5, y_step=0.5, z_step=0.5):
        """
        @param: x_range - (x_min, x_max) by default the min and max of the 3D space
        @param: y_range - (y_min, y_max) by default the min and max of the 3D space
        @param: z_range - (z_min, z_max) by default the min and max of the 3D space
        @param: x_step  - step between samples in x direction
        @param: y_step  - step between samples in y direction
        @param: z_step  - step between samples in z direction
        """
        self.sample_space = []
        self.x_step = x_step
        self.y_step = y_step
        self.z_step = z_step
        self.x_min = x_range[0]
        self.x_max = x_range[1]
        self.y_min = y_range[0]
        self.y_max = y_range[1]
        self.z_min = z_range[0]
        self.z_max = z_range[1]
        for x in np.arange(*self.get_iterable_range(x_range, x_step)):
            for y in np.arange(*self.get_iterable_range(y_range, y_step)):
                for z in np.arange(*self.get_iterable_range(z_range, z_step)):
                    self.sample_space.append(np.array([x, y, z]))
        self.sample_space = np.array(self.sample_space)
    
    def get_sample_space(self):
        """
        Returns current sample space
        """
        return self.sample_space

    def get_iterable_range(self, standard_range, step):
        """
        @param: standard_range: a python range tuple (start, end) with start inclusive, end excluse
        @step: step to sample range
        @return: tuple (start, end, step) where end will include current excluded value 
        """
        return (standard_range[0], standard_range[1] + step, step)

    def get_implicit_function_samples(self, func):
        """
        @param: func - the implicit function to sample
        @return: samples of implicit function across sample space
        """
        self.function_map = {}
        for sample_point in self.sample_space:
            result = func(sample_point)
            x, y, z = sample_point
            self.function_map[(x, y, z)] = result 

        return self.function_map

    def march_cubes(self, condition, implicit_function):
        mesh = VGroup()
        count = 0
        for key in self.function_map:
            print(count)
            if key[0] >= self.x_max or key[1] >= self.y_max or key[2] >= self.z_max:
                continue

            # if count > 1:
            #     break
            marching_cube = self.get_marching_cube(key, implicit_function)
            # cube_mob = self.show_marching_cube(marching_cube)
            # vertex_labels = self.label_cube_vertices(marching_cube)
            # self.wait(0.1)
            
            # self.remove(vertex_labels)

            cube_index = marching_cube.get_cube_index(condition)
            # case_integer = Integer(case).scale(0.4)
            # case_integer.move_to(np.mean(square_corners, axis=0))
            # self.add(case_integer)
            triangles = marching_cube.get_triangles_for_case(cube_index)
            if triangles:
                mesh.add(triangles)
                # self.play(
                #     Create(triangles)
                # )
            # self.remove(cube_mob)
            # print(case)
            count += 1
        return mesh

    def show_marching_cube(self, marching_cube):
        cube = Cube(side_length=marching_cube.get_side_length(), fill_color=PURE_GREEN)
        cube.move_to(np.mean(marching_cube.get_corners(), axis=0))
        self.add(cube)
        return cube

    def label_cube_vertices(self, marching_cube):
        labels = VGroup()
        vertices = marching_cube.get_corners()
        for i in range(len(vertices)):
            label = Integer(i).scale(0.7).move_to(vertices[i] + RIGHT * 0.2)
            labels.add(label)
        self.add(labels)
        return labels

    def get_marching_cube(self, key, implicit_function):
        # Labeling vertices according to this diagram: 
        # http://paulbourke.net/geometry/polygonise/
        v_3 = np.array(key)
        v_2 = np.array([key[0] + self.x_step, key[1], key[2]])
        v_1 = np.array([key[0] + self.x_step, key[1] + self.y_step, key[2]])
        v_0 = np.array([key[0], key[1] + self.y_step, key[2]])
        v_4 = np.array([key[0], key[1] + self.y_step, key[2] + self.z_step])
        v_5 = np.array([key[0] + self.x_step, key[1] + self.y_step, key[2] + self.z_step])
        v_6 = np.array([key[0] + self.x_step, key[1], key[2] + self.z_step])
        v_7 = np.array([key[0], key[1], key[2] + self.z_step])
        return MarchingCube([v_0, v_1, v_2, v_3, v_4, v_5, v_6, v_7], self.function_map, implicit_function)


class Ball(Circle):
    def __init__(self, radius=0.15, position=ORIGIN, **kwargs):
        Circle.__init__(self, radius=radius, ** kwargs)
        self.center = position
        self.velocity = np.array((2, 0, 0))
        self.mass = PI * self.radius ** 2

    ### Only gives scalar values ###
    def get_top(self):
        return self.get_center()[1] + self.radius

    def get_bottom(self):
        return self.get_center()[1] - self.radius

    def get_right_edge(self):
        return self.get_center()[0] + self.radius

    def get_left_edge(self):
        return self.get_center()[0] - self.radius

    ### Gives full vector ###
    def get_top_v(self):
        return self.get_center() + self.radius * UP

    def get_bottom_v(self):
        return self.get_center() + self.radius * DOWN

    def get_right_edge_v(self):
        return self.get_center() + self.radius * RIGHT

    def get_left_edge_v(self):
        return self.get_center() + self.radius * LEFT

    def get_center(self):
        return self.center

    def set_center(self, new_position):
        self.center = new_position

    def set_radius(self, radius):
        self.radius = radius

class Box(Rectangle):
    def __init__(self, **kwargs):
        Rectangle.__init__(self, height=6, width=12, color=GREEN_C, **kwargs)  # Edges

    def get_top(self):
        return self.get_center()[1] + (self.height / 2)

    def get_bottom(self):
        return self.get_center()[1] - (self.height / 2)

    def get_right_edge(self):
        return self.get_center()[0] + (self.width / 2)

    def get_left_edge(self):
        return self.get_center()[0] - (self.width / 2)


class CirclesAndEllipses(Scene):
    def construct(self):
        self.plane = NumberPlane()
        self.add(self.plane)

        self.show_circle_and_ellipse()
    
    def show_circle_and_ellipse(self):
        circle = Circle(radius=1.5).set_color(CONTOUR_COLOR)
        circle.move_to(LEFT * 3.5 + UP * 2)

        circle_center_dot = Dot().move_to(circle.get_center())
        self.play(
            Create(circle)
        )
        center_label_circle = MathTex("(x_0, y_0)").scale(0.8)

        radius_line = DashedLine(circle_center_dot.get_center(), circle_center_dot.get_center() + normalize(UR) * circle.radius)
        radius_label = MathTex("r").scale(0.8)
        radius_label.move_to(radius_line.get_center()).shift(UL * SMALL_BUFF * 1)

        center_label_circle.next_to(circle_center_dot, DOWN, buff=SMALL_BUFF * 2)
        self.play(
            GrowFromCenter(circle_center_dot),
            Write(center_label_circle)
        )

        self.play(
            Write(radius_line),
            Write(radius_label),
        )
        self.wait()

        circle_equation = MathTex("(x - x_0)^2 + (y - y_0)^2  = r^2")
        circle_equation.add_background_rectangle()
        circle_equation.move_to(LEFT * 3.5 + DOWN * 0.7)

        self.play(
            FadeIn(circle_equation)
        )
        self.wait()

        ellipse = Ellipse(width=3.5, height=3).set_color(PINK)
        ellipse.move_to(RIGHT * 3.5 + UP * 2)
        ellipse_center_dot = Dot().move_to(ellipse.get_center())

        center_label_ellipse = MathTex("(x_1, y_1)").scale(0.8)

        center_label_ellipse.next_to(ellipse_center_dot, DOWN, buff=SMALL_BUFF * 2)
        
        self.play(
            Create(ellipse)
        )

        a_line = DashedLine(ellipse.get_center(), ellipse.get_center() + UP * ellipse.height / 2)
        b_line = DashedLine(ellipse.get_center(), ellipse.get_center() + RIGHT * ellipse.width / 2)

        a_label = MathTex("a").scale(0.8).next_to(a_line, LEFT, buff=SMALL_BUFF)
        b_label = MathTex("b").scale(0.8).next_to(b_line, UP, buff=SMALL_BUFF)


        self.play(
            GrowFromCenter(ellipse_center_dot),
            Write(center_label_ellipse)
        )

        self.play(
            Write(a_line),
            Write(b_line)
        )

        self.play(
            Write(a_label),
            Write(b_label)
        )
        self.wait()

        ellipse_equation = MathTex(r"\frac{(x - x_1)^2}{a^2} + \frac{(y - y_1)^2}{b^2} = 1")
        ellipse_equation.add_background_rectangle().move_to(RIGHT * 3.5 + DOWN * 0.7)
        self.play(
            FadeIn(ellipse_equation)
        )
        self.wait()

        self.play(
            ellipse_equation.animate.next_to(circle_equation, DOWN)
        )
        self.wait()

        text_group = VGroup(circle_equation, ellipse_equation)
        surrounding_rect = SurroundingRectangle(text_group)
        implicit_function = Tex("Implicit Function").set_color(YELLOW)
        implicit_function.add_background_rectangle()
        implicit_function.move_to(RIGHT * 3.5 + DOWN * 0.7)
        self.play(
            Create(surrounding_rect),
            FadeIn(implicit_function),
        )
        self.wait()

        implicit_function_def = MathTex("f(x, y) = C").next_to(implicit_function, DOWN)
        implicit_function_def.add_background_rectangle()
        self.play(
            FadeIn(implicit_function_def)
        )
        self.wait()

        explicit_function = Tex(r"Explicit function: $y = f(x)$").scale(0.8)
        explicit_function.add_background_rectangle()
        explicit_function.next_to(implicit_function_def, DOWN).shift(DOWN * SMALL_BUFF * 2)
        self.play(
            FadeIn(explicit_function)
        )
        self.wait()


class DrawFunction(MarchingSquaresUtils):
    def construct(self):
        SIM_TIME = 30
        x_step, y_step = 0.075, 0.075
        sample_space_start_time = time.time()
        
        self.set_sample_space(y_range=(-3, 3), x_step=x_step, y_step=y_step)
        sample_space_end_time = time.time()
        implicit_function = sin_function
        function_map = self.get_implicit_function_samples(implicit_function)
        implicit_function_end_time = time.time()
        condition = lambda x: x >= 1
        contour = self.march_squares(condition, implicit_function)
        marching_squares_end_time = time.time()
        print('Sample space size: {0}'.format(len(self.sample_space)))
        print("--- {0} seconds set_sample_space---".format(sample_space_end_time - sample_space_start_time))
        print("--- {0} seconds get_implicit_function_samples---".format(implicit_function_end_time - sample_space_end_time))
        print("--- {0} seconds march_squares---".format(marching_squares_end_time - implicit_function_end_time))

        self.add(contour)
        self.wait()


class GuessImplicitFunctionValues(MarchingSquaresUtils):
    def construct(self):
        plane = NumberPlane()
        self.play(
            Write(plane),
            run_time=2,
        )
        self.wait()

        # self.contour = self.get_contour(batman_function)

        self.introduce_game()
        self.simplify_game()
        # self.sample_points_randomly(batman_function)

    def introduce_sampling_rule(self):
        sample_dot = Dot().set_color(RED)

        _, x_val, _, y_val, _, sample_val = text_group = VGroup(
            MathTex("f("), 
            DecimalNumber(0.00, num_decimal_places=2, include_sign=True), 
            MathTex(", "),
            DecimalNumber(0.00, num_decimal_places=2, include_sign=True),
            MathTex(") = "),
            DecimalNumber(15.53, num_decimal_places=2, include_sign=True),
        ).arrange(RIGHT)
        text_group[2].shift(DOWN * SMALL_BUFF)
        text_group.add_background_rectangle()
        text_group.next_to(sample_dot, DOWN)

        self.play(
            GrowFromCenter(sample_dot),
            FadeIn(text_group),
        )
        self.wait()

        always(text_group.next_to, sample_dot, DOWN)
        f_always(x_val.set_value, lambda: sample_dot.get_center()[0])
        f_always(y_val.set_value, lambda: sample_dot.get_center()[1])
        f_always(sample_val.set_value, self.sample_scalar)


        path = [
            ORIGIN,
            RIGHT * 2 + UP * 0.5,
            RIGHT * 3 + UP * 2,
            RIGHT * 1 + UP * 3,
            RIGHT * 1.5 + UP * 1,
            LEFT * 1.5 + UP * 0.5,
            LEFT * 3 + UP * 2, 
            LEFT * 2.5 + DOWN * 0.5,
            LEFT * 0.5 + DOWN * 2.5,
            RIGHT * 1 + DOWN * 1.5,
            RIGHT * 2,
        ]

        path_group = VGroup().set_points_smoothly(*[path])

        self.play(
            MoveAlongPath(sample_dot, path_group),
            run_time=8,
            rate_func=linear,
        )
        self.wait()

        text_group.clear_updaters()

        self.play(
            FadeOut(text_group),
            FadeOut(sample_dot)
        )
        self.wait()

    def introduce_game(self):
        mysterious_function = Tex(r"$f(x, y) =$ ?")
        mysterious_function.add_background_rectangle()
        mysterious_function.move_to(RIGHT * 5.7 + DOWN * 2.9)
        self.play(
            FadeIn(mysterious_function)
        )
        self.wait()

        self.introduce_sampling_rule()


        goal = Tex(r"Find all points where $f(x, y) = 1$").scale(0.8)
        goal.add_background_rectangle()
        goal.next_to(mysterious_function, DOWN, aligned_edge=RIGHT)

        self.play(
            FadeIn(goal)
        )
        self.wait()

        self.add_foreground_mobjects(mysterious_function, goal)

        labels = VGroup()

        points = [(-3.5, -3), (5, 2), (-0.4, 2.2)]
        pos_neg = [False, True, True]
        contour_mask = [False, False, True]
        i = 0
        for x, y in points:
            label = self.label_point(x, y, positive=pos_neg[i], contour_value=contour_mask[i])
            labels.add(label)
            i += 1

        self.play(
            FadeOut(labels),
            FadeOut(goal),
            FadeOut(mysterious_function)
        )
        self.wait()

    def sample_scalar(self, positive=True):
        value = np.random.uniform(4, 25)
        if not positive:
            return -value + np.random.uniform(0, 3)
        return value

    def label_point(self, x, y, positive=True, contour_value=True):
        point = np.array([x, y, 0])
        value = np.round(self.sample_scalar(positive=positive), 1)
        threshold = False
        threshold_value = 1.1
        if contour_value:
            value = threshold_value
        if value < 3 and value > 1:
            value = threshold_value
            threshold = True
        label = MathTex(r"f({0}, {1}) = {2}".format(x, y, value))
        label.scale(0.8)
        dot = Dot().move_to(point)
        # opposite of convention because of way batman function is defined
        if value > 1 and value != threshold_value:
            dot.set_color(INSIDE_COLOR)
        elif value == threshold_value:
            dot.set_color(YELLOW)
        else:
            dot.set_color(OUTSIDE_COLOR)
        label.add_background_rectangle()
        label.next_to(dot, DOWN)

        self.play(
            GrowFromCenter(dot)
        )
        self.wait()
        self.play(
            FadeIn(label)
        )
        self.wait()

        close_enough = Tex("Close enough to target value").scale(0.7)
        close_enough.add_background_rectangle()
        if threshold:
            close_enough.next_to(dot, UP)
            self.play(
                FadeIn(close_enough)
            )
            self.wait()

            basically_impossible = Tex(r"Finding these points randomly" + "\\\\" + "is extremely unlikely").scale(0.7)
            
            basically_impossible.add_background_rectangle()
            basically_impossible.next_to(dot, UP)
            self.play(
                Transform(close_enough, basically_impossible)
            )
            self.wait()

            return VGroup(dot, label, close_enough)
              
        return VGroup(dot, label)

    def simplify_game(self):
        # Note, we are technically switching this around
        # to keep things consistent with later definitions
        # In principle, it doesn't matter
        greater_case = Tex(r"If $f(x, y) > 1 \rightarrow$")
        green_dot = Dot(color=INSIDE_COLOR)
        greater_group = VGroup(greater_case, green_dot).arrange(RIGHT)

        less_case = Tex(r"If $f(x, y) < 1 \rightarrow$")
        blue_dot = Dot(color=OUTSIDE_COLOR)
        less_group = VGroup(less_case, blue_dot).arrange(RIGHT)

        VGroup(less_group, greater_group).arrange(DOWN).move_to(LEFT * 3.5 + UP * 2)
        self.play(
            FadeIn(greater_group),
            FadeIn(less_group)
        )

        self.wait()

        self.play(
            FadeOut(greater_group),
            FadeOut(less_group)
        )
        self.wait()

    def sample_points_randomly(self, implicit_function):
        first_thousand = self.get_set_of_points(1000, implicit_function)
        to_ten_thousand = self.get_set_of_points(9000, implicit_function)
        to_twenty_thousand = self.get_set_of_points(10000, implicit_function)
        

        self.play(
            LaggedStartMap(GrowFromCenter, first_thousand),
            run_time=2
        )
        self.wait()

        self.play(
            LaggedStartMap(GrowFromCenter, to_ten_thousand),
            run_time=3
        )
        self.wait()

        self.play(
            LaggedStartMap(GrowFromCenter, to_twenty_thousand),
            run_time=3
        )
        self.wait()

        num_points = 2000
        contour_points = VGroup()
        for _ in range(num_points):
            point = self.sample_point_from_contour(self.contour)
            if self.filter_point(point):
                continue
            dot = Dot(radius=0.05).move_to(point).set_color(YELLOW)
            contour_points.add(dot)

        self.play(
            LaggedStartMap(GrowFromCenter, contour_points),
            run_time=2,
        )
        self.wait()


    def get_set_of_points(self, num_points, implicit_function):
        dots = VGroup()
        radius = 0.05
        for _ in range(num_points):
            point = get_random_point_in_frame()
            value = implicit_function(point)
            dot = Dot(radius=radius).move_to(point)
            # opposite of convention because of way batman function is defined
            if value > 1:
                if self.change_blue_point(point):
                    dot.set_color(INSIDE_COLOR)
                else:
                    dot.set_color(OUTSIDE_COLOR)                
            else:
                if self.change_green_point(point):
                    dot.set_color(OUTSIDE_COLOR)
                else:
                    dot.set_color(INSIDE_COLOR)

            dots.add(dot)

        return dots

    def change_blue_point(self, point):
        x, y, _ = point
        general_internal_bb = x > -0.95 and x < 0.95 and y > -0.5 and y < 2 
        fine_grained_bb_1 = x > -0.8 and x < 0.8 and y > 2 and y < 2.25 
        left_triangle = Polygon(
            np.array([-2.4, 1, 0]),
            np.array([-3, 1, 0]),
            np.array([-3, 2, 0])
        )
        right_triangle = Polygon(
            np.array([2.4, 1, 0]),
            np.array([3, 1, 0]),
            np.array([3, 2, 0])
        )

        left_quad = Polygon(
            np.array([-0.5, 2.25, 0]),
            np.array([-0.88, 2.25, 0]),
            np.array([-0.8, 2.8, 0]),
            np.array([-0.7, 2.8, 0]),
        )

        right_quad = Polygon(
            np.array([0.5, 2.25, 0]),
            np.array([0.88, 2.25, 0]),
            np.array([0.8, 2.8, 0]),
            np.array([0.7, 2.8, 0]),
        )

        return general_internal_bb or fine_grained_bb_1 or \
            check_point_inside(point, left_triangle) or \
            check_point_inside(point, right_triangle) or \
            check_point_inside(point, left_quad) or \
            check_point_inside(point, right_quad)

    def change_green_point(self, point):
        x, y, _ = point
        general_left_bb = x < -1 and x > -2.5 and y > 1 and y < 3
        general_right_bb = x > 1 and x < 2.5 and y > 1 and y < 3
        gen_vert_bounds = y > 2.9 or y < -2.65
        one_off_quadrants = (x < -2.5 and x > -3 and y > 2 and y < 3) or \
            (x > 2.5 and x < 3 and y > 2 and y < 3) or \
            (x > -1.1 and x < -0.5 and y < -1.75 and y > -3) or \
            (x < 1.1 and x > 0.5 and y < -1.75 and y > -3) or \
            (x > 0 and x < 0.5 and y < -2.5 and y > -3) or \
            (x > -1 and x < 1 and y > 2.25 and y < 3)


        left_triangle = Polygon(
            np.array([-4, -2.5, 0]),
            np.array([-7.2, -1.75, 0]),
            np.array([-7.2, -2.5, 0]),
        )
        right_triangle = Polygon(
            np.array([4, -2.5, 0]),
            np.array([7.2, -1.75, 0]),
            np.array([7.2, -2.5, 0]),
        )

        top_left_triangle = Polygon(
            np.array([-2.9, 2, 0]),
            np.array([-2.5, 2, 0]),
            np.array([-2.5, 1.2, 0]),
        )

        top_right_triangle = Polygon(
            np.array([2.9, 2, 0]),
            np.array([2.45, 2, 0]),
            np.array([2.45, 1.1, 0]),
        )
        return general_left_bb or general_right_bb or gen_vert_bounds or \
            check_point_inside(point, left_triangle) or \
            check_point_inside(point, right_triangle) or \
            check_point_inside(point, top_left_triangle) or \
            check_point_inside(point, top_right_triangle) or \
            one_off_quadrants

    def filter_point(self, point):
        # Filter out points that are noisy in batman curve
        # Not ideal solution, more of a hacky workaround
        x, y, _ = point
        vertical_and_horizontal_bounds = y < -2.7 or y > 3 or x > -1 and x < 1 and y > -1 and y < 1
        close_y_3 = abs(y - 3) < 0.1
        horiz_line_bottom = x < -4.3 and abs(y + 2.5) < 0.1 or x > 4.3 and abs(y + 2.5) < 0.1
        vert_lines = y < -1.7 and (abs(x) < 0.05 or abs(x + 1) < 0.1 or abs(x - 1) < 0.1 or abs(x + 0.85) < 0.05 or abs(x - 0.7) < 0.05) 
        one_off_quadrants = (x > 1 and y > 2 and x < 2 and y < 3) or \
            (x < -3 and x > -4 and y > 1 and y < 2) or \
            (x < -6.6 and y < -1 and y > -1.5) or \
            (y < -1.5 and y > -2.1 and x < -6) or \
            (x > 6.52 and y < -1.05 and y > -1.5)
        one_off_points = [
        np.array([-0.8, 1.25, 0]), np.array([-2.85, 1.1, 0]), 
        np.array([-0.1, -1.8, 0]), np.array([-2.85, 2.85, 0]),
        np.array([-5.25, -2.1, 0]), np.array([-5.55, -2.05, 0]),
        np.array([-5.8, -2, 0]), np.array([-6.45, -1.42, 0]),
        np.array([6.25, -1.5, 0]), np.array([6.35, -1.4, 0]),
        np.array([-0.4, -2.35, 0]), np.array([0.75, 1.1, 0]),
        np.array([0.85, 1.3, 0]),
        ]
        is_close_to_one_off_points = any([np.linalg.norm(point - p) < 0.1 for p in one_off_points])
        other_vert_lines = y < 2 and y > 0.5 and abs(x - 3) < 0.05 or y < 1.1 and y > 0.5 and abs(x - 2.75) < 0.1
        return vertical_and_horizontal_bounds or close_y_3 or \
            horiz_line_bottom  or vert_lines or \
            one_off_quadrants or is_close_to_one_off_points or \
            other_vert_lines
        

    def get_contour(self, implicit_function):
        x_step, y_step = 0.075, 0.075
        sample_space_start_time = time.time()
        
        self.set_sample_space( x_step=x_step, y_step=y_step)
        sample_space_end_time = time.time()
        function_map = self.get_implicit_function_samples(implicit_function)
        implicit_function_end_time = time.time()
        condition = lambda x: x >= 1
        contour = self.march_squares(condition, implicit_function)
        marching_squares_end_time = time.time()
        print('Sample space size: {0}'.format(len(self.sample_space)))
        print("--- {0} seconds set_sample_space---".format(sample_space_end_time - sample_space_start_time))
        print("--- {0} seconds get_implicit_function_samples---".format(implicit_function_end_time - sample_space_end_time))
        print("--- {0} seconds march_squares---".format(marching_squares_end_time - implicit_function_end_time))
        return contour

    def sample_point_from_contour(self, contour):
        index = np.random.randint(len(contour))
        set_of_lines = contour[index]
        nested_index = np.random.choice(list(range(len(set_of_lines))))
        return set_of_lines[nested_index].get_start()

class ImplicitFunctionCases(Scene):
    def construct(self):
        self.show_cases()
        
    def show_cases(self):
        A, B = self.show_first_case()
        A, B = self.show_second_case(A, B)
        A, B, contour = self.show_third_case(A, B)
        f_c = self.show_root_finding(A, B)
        self.make_note_about_convention(f_c)
        self.show_key_takeaway()
        self.play(
            ApplyWave(contour)
        )
        self.wait()

    def make_note_about_convention(self, f_c):
        self.play(
            f_c.animate.shift(UP * 2)
        )
        self.wait()

        inside_convention = Tex(r"$f(B) > 1 \Rightarrow $ point $B$ inside contour").to_edge(LEFT * 2).shift(DOWN * 2.2)
        outside_convention = Tex(r"$f(A) < 1 \Rightarrow $ point $A$ outside contour").next_to(inside_convention, DOWN, aligned_edge=LEFT)
        inside_convention[0][-13:-7].set_color(INSIDE_COLOR)
        outside_convention[0][-14:-7].set_color(OUTSIDE_COLOR)

        self.play(
            FadeIn(inside_convention)
        )
        self.wait()
        self.play(
            FadeIn(outside_convention)
        )
        self.wait()

        self.play(
            FadeOut(inside_convention),
            FadeOut(outside_convention)
        )
        self.wait()

    def show_key_takeaway(self):
        key_takeaway = Tex("Key Idea").to_edge(LEFT * 2).shift(DOWN * 1.8)
        underline = Underline(key_takeaway)

        self.play(
            Write(key_takeaway),
            Write(underline)
        )

        self.wait()

        fact = Tex(
            r"Given $f(A) < 1$ and $f(B) > 1$, we can find" + "\\\\",
            r"$C$ such that $f(C) \approx 1$ using root-finding."
        ).scale(0.8)

        fact[0].to_edge(LEFT * 2)
        fact[1].to_edge(LEFT * 2)

        fact[0][7].set_color(OUTSIDE_COLOR)
        fact[0][16].set_color(INSIDE_COLOR)

        fact[1][0].set_color(CONTOUR_COLOR)
        fact[1][11].set_color(CONTOUR_COLOR)

        fact.next_to(underline, DOWN, aligned_edge=LEFT)

        self.play(
            FadeIn(fact)
        )
        self.wait()




    def show_first_case(self):
        pointA = Dot().set_color(INSIDE_COLOR)
        labelA = MathTex("A")
        fA = MathTex("f(A) > 1")

        A_group = VGroup(labelA, pointA, fA).arrange(DOWN).move_to(LEFT * 3)

        pointB = Dot().set_color(INSIDE_COLOR)
        labelB = MathTex("B")
        fB = MathTex("f(B) > 1")

        B_group = VGroup(labelB, pointB, fB).arrange(DOWN).move_to(RIGHT * 3)

        # self.add(A_group, B_group)

        self.play(
            GrowFromCenter(pointA),
            GrowFromCenter(pointB),
            Write(labelA),
            Write(labelB),
        )
        self.wait()

        self.play(
            Write(fA),
            Write(fB),
        )
        self.wait()

        self.add_foreground_mobjects(A_group, B_group)

        first_contour = self.get_smooth_contour_around_points(A_group, B_group)

        self.play(
            DrawBorderThenFill(first_contour)
        )
        self.wait()

        second_contour = self.get_smooth_contour_around_points(A_group, B_group, seed=1)

        self.play(
            ReplacementTransform(first_contour, second_contour)
        )
        self.wait()

        self.play(
            FadeOut(second_contour)
        )

        return A_group, B_group

    def get_smooth_contour_around_points(self, A, B, seed=0):
        np.random.seed(seed)
        directions_A = [
        LEFT, UL, UP, UR,
        DR, DOWN, DL, 
        ]

        A_region_points = [A.get_center() + v * np.random.uniform(1, 2.5) for v in directions_A]

        directions_B = [
        UL, UP, UR, RIGHT,
        DR, DOWN, DL
        ]

        B_region_points = [B.get_center() + v * np.random.uniform(1, 2.5) for v in directions_B]

        contour_points = A_region_points[:4] + B_region_points + A_region_points[4:] + [A_region_points[0]]

        contour = VGroup()
        contour.set_points_smoothly(*[contour_points])
        contour.set_stroke(color=YELLOW)
        contour.set_fill(color=YELLOW, opacity=0.3)

        #set back to default seed
        np.random.seed(0)

        return contour

    def show_second_case(self, A, B):
        pointA = Dot().set_color(OUTSIDE_COLOR)
        labelA = MathTex("A")
        fA = MathTex("f(A) < 1")

        A_group = VGroup(labelA, pointA, fA).arrange(DOWN).move_to(LEFT * 2.5 + DOWN * 2.5)

        pointB = Dot().set_color(OUTSIDE_COLOR)
        labelB = MathTex("B")
        fB = MathTex("f(B) < 1")

        B_group = VGroup(labelB, pointB, fB).arrange(DOWN).move_to(RIGHT * 2.5 + DOWN * 2)


        self.play(
            ReplacementTransform(A, A_group),
            ReplacementTransform(B, B_group)
        )
        self.wait()

        first_contour = self.get_second_case_contour_1()
        self.play(
            DrawBorderThenFill(first_contour)
        )
        self.wait()

        second_contour = self.get_second_case_contour_2()
        self.play(
            ReplacementTransform(first_contour, second_contour)
        )
        self.wait()

        self.play(
            FadeOut(second_contour)
        )
        return A_group, B_group

    def get_second_case_contour_1(self):
        contour_points = [
        LEFT * 7.2,
        LEFT * 5 + DOWN * 1,
        LEFT * 3 + DOWN * 0.5,
        LEFT * 1 + DOWN * 2,
        RIGHT * 1 + DOWN,
        RIGHT * 3 + DOWN * 0.2,
        RIGHT * 5 + DOWN * 1.5,
        RIGHT * 7.2 + DOWN * 0.4,
        RIGHT * 7.2 + UP * 4.1,
        LEFT * 7.2 + UP * 4.1,
        LEFT * 7.2,
        ]
        contour = VGroup()
        contour.set_points_smoothly(*[contour_points])
        contour.set_stroke(color=YELLOW)
        contour.set_fill(color=YELLOW, opacity=0.3)
        return contour

    def get_second_case_contour_2(self):
        contour_points = [
        LEFT * 6 + DOWN * 4.1,
        LEFT * 5 + DOWN * 0.5,
        LEFT * 2.5 + DOWN * 0.9,
        DOWN * 3,
        RIGHT * 0.5 + DOWN,
        RIGHT * 1 + UP * 1.5,
        RIGHT * 3 + UP * 4.1,
        LEFT * 7.2 + UP * 4.1,
        LEFT * 7.2 + DOWN * 4.1,
        LEFT * 6 + DOWN * 4.1,
        ]
        contour = VGroup()
        contour.set_points_smoothly(*[contour_points])
        contour.set_stroke(color=YELLOW)
        contour.set_fill(color=YELLOW, opacity=0.3)
        return contour

    def show_third_case(self, A, B):
        pointA = Dot().set_color(OUTSIDE_COLOR)
        labelA = MathTex("A")
        fA = MathTex("f(A) < 1")

        A_group = VGroup(labelA, pointA, fA).arrange(DOWN).move_to(LEFT * 4 + UP * 0.5)
        # VGroup(A_group[0], A_group[2]).arrange(DOWN).next_to(pointA, LEFT)

        pointB = Dot().set_color(INSIDE_COLOR)
        labelB = MathTex("B")
        fB = MathTex("f(B) > 1")

        B_group = VGroup(labelB, pointB, fB).arrange(DOWN).move_to(RIGHT * 4 + DOWN * 0.5)
        # VGroup(B_group[0], B_group[2]).arrange(DOWN).next_to(pointB, RIGHT)
        self.play(
            ReplacementTransform(A, A_group),
            ReplacementTransform(B, B_group)
        )
        self.wait()

        contour = self.get_third_case_contour()
        self.play(
            DrawBorderThenFill(contour)
        )
        self.wait()

        return A_group, B_group, contour

    def get_third_case_contour(self):
        contour_points = [
        LEFT * 2 + UP * 4.1,
        LEFT * 1 + UP * 2,
        RIGHT * 2.5 + DOWN * 1,
        RIGHT * 4.5 + DOWN * 4.1, 
        RIGHT * 7.2 + DOWN * 4.1,
        RIGHT * 7.2 + UP * 4.1,
        LEFT * 2 + UP * 4.1,
        ]
        contour = VGroup()
        contour.set_points_smoothly(*[contour_points])
        contour.set_stroke(color=YELLOW)
        contour.set_fill(color=YELLOW, opacity=0.3)
        return contour

    def show_root_finding(self, A, B):
        self.add_foreground_mobjects(A[1], B[1])
        self.wait()
        line_between_points = Line(A[1].get_center(), B[1].get_center())
        self.play(
            Create(line_between_points),
            run_time=2
        )
        self.wait()

        C_label = MathTex("C")
        C_dot = Dot().set_color(OUTSIDE_COLOR)
        f_c = Tex(r"$f(C) \approx 1$?").scale(1.2)
        f_c.move_to(DOWN * 3)

        C_dot.move_to(line_between_points.point_from_proportion(1/2))
        C_label.next_to(C_dot, UP)

        self.play(
            GrowFromCenter(C_dot),
            Write(C_label)
        )

        self.play(
            Write(f_c)
        )
        self.wait()

        always(C_label.next_to, C_dot, UP)

        self.play(
            C_dot.animate.move_to(line_between_points.point_from_proportion(3/4)).set_color(INSIDE_COLOR),
        )
        self.wait()

        self.play(
            C_dot.animate.move_to(line_between_points.point_from_proportion(5/8)).set_color(OUTSIDE_COLOR),
        )
        self.wait()

        self.play(
            C_dot.animate.move_to(line_between_points.point_from_proportion(11/16)).set_color(OUTSIDE_COLOR),
        )
        self.wait()

        self.play(
            C_dot.animate.move_to(line_between_points.point_from_proportion(23/32)).set_color(OUTSIDE_COLOR),
        )
        self.play(
            Flash(C_dot),
            C_dot.animate.set_color(YELLOW),
        )
        self.wait()

        return f_c


class MarchingSquaresCases(Scene):
    def construct(self):
        all_cases_in_grid = self.introduce_cases()

        self.organize_cases(all_cases_in_grid)

    def introduce_cases(self):
        cases = list(itertools.product([0, 1], repeat=4))
        all_squares = [self.get_case(case) for case in cases]
        all_cases_in_grid = self.arrange_squares_in_grid(all_squares)
        all_cases_in_grid.move_to(ORIGIN)
        print(len(all_cases_in_grid[0]))
        print(len(all_cases_in_grid[1]))
        print(len(all_cases_in_grid[2]))
        print(len(all_cases_in_grid[3]))
        self.play(
            FadeIn(all_cases_in_grid)
        )
        self.wait()

        self.highlight_cases_seen(all_cases_in_grid)

        self.set_contour_lines()
        case_10 = self.get_case_contour(all_cases_in_grid[1][0], self.get_line_proportions_for_case(1, 0))
        case_12 = self.get_case_contour(all_cases_in_grid[1][2], self.get_line_proportions_for_case(1, 2))
        self.all_case_contours = {(1, 0): case_10, (1, 2): case_12}
        self.animate_showing_case([case_10, case_12])

        self.show_adjustment_of_points(case_10, case_12)

       

        return all_cases_in_grid

    def organize_cases(self, all_cases_in_grid):
        # the add_to_back calls mutate the original grid

        self.set_contour_polygons(all_cases_in_grid)
        self.all_case_polygons = {}
        self.all_case_polygons[(1, 0)] = self.get_polygon_for_case(all_cases_in_grid, 1, 0)
        self.all_case_polygons[(1, 2)] = self.get_polygon_for_case(all_cases_in_grid, 1, 2)
        self.all_case_polygons[(3, 3)] = self.get_polygon_for_case(all_cases_in_grid, 3, 3)

        for i in range(4):
            for j in range(4):
                self.add_foreground_mobject(all_cases_in_grid[i][j])

        self.add_foreground_mobject(self.all_case_contours[(1, 0)])
        self.add_foreground_mobject(self.all_case_contours[(1, 2)])

        self.play(
            *[FadeIn(p) for p in self.all_case_polygons.values()]
        )
        self.wait()

        to_remove = []
        for i in range(4):
            for j in range(4):
                if (i, j) not in [(0, 0), (3, 3)]:
                    to_remove.append(all_cases_in_grid[i][j])
        

        new_group_no_contour = VGroup(
            all_cases_in_grid[0][0].copy(),
            all_cases_in_grid[3][3].copy(),
        ).arrange(DOWN).move_to(LEFT * 5)

        self.play(
            *[FadeOut(case) for case in to_remove],
            FadeOut(self.all_case_contours[(1, 0)]),
            FadeOut(self.all_case_contours[(1, 2)]),
            FadeOut(self.all_case_polygons[(1, 0)]),
            FadeOut(self.all_case_polygons[(1, 2)]),
            all_cases_in_grid[0][0].animate.move_to(new_group_no_contour[0].get_center()),
            all_cases_in_grid[3][3].animate.move_to(new_group_no_contour[1].get_center()),
            self.all_case_polygons[(3, 3)].animate.move_to(new_group_no_contour[1].get_center()),
        )
        self.wait()

        self.show_column(all_cases_in_grid, [(1, 0), (2, 0), (0, 1), (0, 2)], LEFT * 2.5)

        self.show_column(all_cases_in_grid, [(1, 2), (3, 0), (2, 1), (0, 3)], ORIGIN)

        contours, polygons = self.show_column(all_cases_in_grid, [(1, 1), (2, 2)], RIGHT * 2.5)

        self.show_column(all_cases_in_grid, [(3, 2), (3, 1), (2, 3), (1, 3)], RIGHT * 5)
    
        self.highlight_edge_cases(all_cases_in_grid, contours, polygons)
    def show_column(self, all_cases_in_grid, keys, position):
        column = VGroup(*[all_cases_in_grid[i][j] for i, j in keys])
        column.arrange(DOWN).move_to(position)
        contours = []
        polygons = []
        for i, j in keys:
            contours.append(
                self.get_case_contour(
                    all_cases_in_grid[i][j], 
                    self.get_line_proportions_for_case(i, j)
                )
            )
            polygons.append(
                self.get_polygon_for_case(all_cases_in_grid, i, j)
            )

        self.play(
            FadeIn(column)
        )
        self.add_foreground_mobject(column)
        self.wait()

        self.animate_showing_case(contours)
        self.play(
            *[FadeIn(p) for p in polygons]
        )
        self.wait()
        return contours, polygons

    def highlight_edge_cases(self, all_cases_in_grid, contours, polygons):
        edge_cases = VGroup(all_cases_in_grid[1][1], all_cases_in_grid[2][2])
        surround_rect = SurroundingRectangle(edge_cases).set_color(ORANGE)

        self.play(
            Create(surround_rect)
        )
        self.wait()

        self.play(
            *[Indicate(c) for c in contours],
            *[Indicate (p) for p in polygons],
        )
        self.wait()

        dot1 = Dot().set_color(INSIDE_COLOR).move_to(edge_cases[0].get_center())
        dot2 = Dot().set_color(INSIDE_COLOR).move_to(edge_cases[1].get_center())

        self.play(
            GrowFromCenter(dot1),
            GrowFromCenter(dot2),
        )
        self.wait()

        shift_amount = np.linalg.norm(edge_cases[0].get_center() - edge_cases[1].get_center())
        edge_case_1 = edge_cases[0].copy().shift(UP * shift_amount)
        edge_case_2 = edge_cases[1].copy().shift(DOWN * shift_amount)

        self.play(
            TransformFromCopy(edge_cases[0], edge_case_1),
            TransformFromCopy(edge_cases[1], edge_case_2),
        )
        self.wait()

        dot3 = Dot().set_color(OUTSIDE_COLOR).move_to(edge_case_1.get_center())
        dot4 = Dot().set_color(OUTSIDE_COLOR).move_to(edge_case_2.get_center())
        self.play(
            FadeOut(surround_rect),
            GrowFromCenter(dot3),
            GrowFromCenter(dot4),
        )
        self.wait()

        case_1_contour = self.get_case_contour(edge_case_1, [(1/8, 3/8), (5/8, 7/8)])
        case_2_contour = self.get_case_contour(edge_case_2, [(7/8, 1/8), (3/8, 5/8)])

        self.animate_showing_case([case_1_contour, case_2_contour])

        polygons_for_case_1 = VGroup(*[self.get_polygon_for_case(all_cases_in_grid, i, j, case=edge_case_1) for i, j in [(4, 0), (4, 1)]])
        polygons_for_case_2 = VGroup(*[self.get_polygon_for_case(all_cases_in_grid, i ,j, case=edge_case_2) for i, j in [(4, 2), (4, 3)]])

        self.play(
            FadeIn(polygons_for_case_1),
            FadeIn(polygons_for_case_2)
        )
        self.wait()

    def set_contour_lines(self):
        self.index_to_line_proportion = {
        (0, 0): [],
        (0, 1): [(5/8, 7/8)],
        (0, 2): [(3/8, 5/8)],
        (0, 3): [(3/8, 7/8)],
        (1, 0): [(1/8, 3/8)],
        (1, 1): [(1/8, 7/8), (3/8, 5/8)],
        (1, 2): [(1/8, 5/8)],
        (1, 3): [(1/8, 7/8)],
        (2, 0): [(7/8, 1/8)],
        (2, 1): [(1/8, 5/8)],
        (2, 2): [(1/8, 3/8), (5/8, 7/8)],
        (2, 3): [(1/8, 3/8)],
        (3, 0): [(3/8, 7/8)],
        (3, 1): [(3/8, 5/8)],
        (3, 2): [(5/8, 7/8)],
        (3, 3): []
        }

    def set_contour_polygons(self, all_cases_in_grid):
        self.index_to_proportitions = {
        (0, 0): [],
        (0, 1): [5/8, 6/8, 7/8],
        (0, 2): [3/8, 4/8, 5/8],
        (0, 3): [3/8, 4/8, 6/8, 7/8],
        (1, 0): [1/8, 2/8, 3/8],
        (1, 1): [1/8, 2/8, 3/8, 5/8, 6/8, 7/8],
        (1, 2): [1/8, 2/8, 4/8, 5/8],
        (1, 3): [1/8, 2/8, 4/8, 6/8, 7/8],
        (2, 0): [7/8, 0, 1/8],
        (2, 1): [1/8, 5/8, 6/8, 0],
        (2, 2): [1/8, 3/8, 4/8, 5/8, 7/8, 0],
        (2, 3): [1/8, 3/8, 4/8, 6/8, 0],
        (3, 0): [0, 2/8, 3/8, 7/8],
        (3, 1): [0, 2/8, 3/8, 5/8, 6/8],
        (3, 2): [0, 2/8, 4/8, 5/8, 7/8],
        (3, 3): [0, 2/8, 4/8, 6/8],
        # these are edge cases
        # begin case 1
        (4, 0): [1/8, 2/8, 3/8], 
        (4, 1): [5/8, 6/8, 7/8],
        # begin case 2
        (4, 2): [7/8, 0, 1/8],
        (4, 3): [3/8, 4/8, 5/8],
        }
    
    def get_polygon_for_case(self, all_cases_in_grid, i, j, case=None):
        if not case:
            case = all_cases_in_grid[i][j]
        square, _ = case
        proporitions = self.index_to_proportitions[(i, j)]
        if not proporitions:
            return VGroup()
        polygon = Polygon(*[square.point_from_proportion(p) for p in proporitions])
        polygon.set_stroke(width=0)
        polygon.set_fill(color=CONTOUR_COLOR, opacity=0.5)
        return polygon

    def get_line_proportions_for_case(self, i, j):
        return self.index_to_line_proportion[(i, j)]

    def animate_showing_case(self, cases):
        all_anims = []
        self.add_foreground_mobjects(*cases)
        for case in cases:
            lines, dots = case
            all_anims.extend(
                [Create(line) for line in lines] + [GrowFromCenter(dot) for dot in dots]
            )
        
        self.play(
            *all_anims
        )
        self.wait()

    def show_adjustment_of_points(self, case_10, case_12):
        line10, dots10 = case_10
        line12, dots12 = case_12

        line10[0].add_updater(lambda line: line.become(Line(dots10[0].get_center(), dots10[1].get_center()).set_color(CONTOUR_COLOR)))
        self.add(line10[0])
        line12[0].add_updater(lambda line: line.become(Line(dots12[0].get_center(), dots12[1].get_center()).set_color(CONTOUR_COLOR)))
        self.add(line12[0])
       

        self.play(
            dots10[0].animate.shift(RIGHT * 0.5),
            dots12[0].animate.shift(LEFT * 0.5),
        )

        self.play(
            dots10[1].animate.shift(DOWN * 0.5),
            dots12[1].animate.shift(RIGHT * 0.5),
        )

        self.play(
            dots10[0].animate.shift(LEFT * 0.5),
            dots12[0].animate.shift(RIGHT * 0.5),
            dots10[1].animate.shift(UP * 0.5),
            dots12[1].animate.shift(LEFT * 0.5),
        )
        self.wait()
        line10.clear_updaters()
        line12.clear_updaters()

    def highlight_cases_seen(self, all_cases_in_grid, color=YELLOW):
        self.play(
            ShowPassingFlash(
                all_cases_in_grid[0][0][0].copy().set_color(color),
                time_width=0.5,
            ),
            ShowPassingFlash(
                all_cases_in_grid[3][3][0].copy().set_color(color),
                time_width=0.5,
            ),
            ShowPassingFlash(
                all_cases_in_grid[1][0][0].copy().set_color(color),
                time_width=0.5,
            ),
            ShowPassingFlash(
                all_cases_in_grid[1][2][0].copy().set_color(color),
                time_width=0.5,
            ),
            run_time=2
        )

        self.play(
            ShowPassingFlash(
                all_cases_in_grid[0][0][0].copy().set_color(color),
                time_width=0.5,
            ),
            ShowPassingFlash(
                all_cases_in_grid[3][3][0].copy().set_color(color),
                time_width=0.5,
            ),
            ShowPassingFlash(
                all_cases_in_grid[1][0][0].copy().set_color(color),
                time_width=0.5,
            ),
            ShowPassingFlash(
                all_cases_in_grid[1][2][0].copy().set_color(color),
                time_width=0.5,
            ),
            run_time=2
        )

    def get_case(self, case):
        square = Square(side_length=1.5).set_stroke(color=BLUE_D)
        dot_corners = VGroup()
        for bit, corner in zip(case, square.get_vertices()):
            dot = Dot().move_to(corner).set_color(OUTSIDE_COLOR)
            if bit:
                dot.set_color(INSIDE_COLOR)
            dot_corners.add(dot)

        return VGroup(square, dot_corners)

    def arrange_squares_in_grid(self, squares):
        rows = []
        for i in range(0, len(squares), 4):
            rows.append(VGroup(*squares[i:i+4]).arrange(RIGHT, buff=1.5))

        return VGroup(*rows).arrange(DOWN)

    def get_case_contour(self, case, proportions):
        square = case[0]
        lines = VGroup()
        dots = []
        polygon_points = []
        for p1, p2 in proportions:
            start_dot = Dot().set_color(CONTOUR_COLOR)
            end_dot = start_dot.copy()
            start = square.point_from_proportion(p1)
            end = square.point_from_proportion(p2)
            start_dot.move_to(start)
            end_dot.move_to(end)
            line = Line(start, end).set_color(CONTOUR_COLOR)
            lines.add(line)
            dots.extend([start_dot, end_dot])
        return VGroup(lines, VGroup(*dots))

class LerpImprovement(Scene):
    def construct(self):
        dots, label_classes, title, case_square = self.remind_big_picture()
        self.motivate_lerp(case_square, dots, label_classes)
    
    def remind_big_picture(self):
        x_step, y_step = 0.5, 0.5
        self.plane = NumberPlane(x_range=(- 7.111111111111111, 7.111111111111111, x_step), y_range=(- 4.0, 4.0, y_step),)
        self.play(
            Write(self.plane),
            run_time=2
        )
        self.wait()

        square = Square(side_length=0.5).move_to(LEFT * 0.75 + DOWN * 0.75)

        self.play(
            Create(square)
        )
        self.wait()

        case_square = Square(side_length=4).set_stroke(color=BLUE_D)

        self.play(
            ReplacementTransform(square, case_square),
            FadeOut(self.plane)
        )
        self.wait()

        dots, label_classes, title = self.show_dual_root_finding(case_square)

        return dots, label_classes, title, case_square

    def show_dual_root_finding(self, case_square):
        dots = VGroup(*[Dot().move_to(corner) for corner in case_square.get_vertices()])

        label_indices = [(0, 0), (1, 0), (0, 1), (1, 1)]
        labels = VGroup(*[MathTex("f(x_{0}, y_{1})".format(i, j)) for i, j in label_indices])
        for i, corner in enumerate(self.get_ordered_corners(case_square)):
            unit_v_from_center = normalize(corner - case_square.get_center())
            labels[i].scale(0.9).move_to(corner + unit_v_from_center * 0.7)

        self.play(
            *[GrowFromCenter(dot) for dot in dots]
        )

        self.play(
            *[Write(label) for label in labels]
        )
        self.wait()
        gt_lt = [">", "<"]
        label_classes = VGroup(*[MathTex("f(x_{0}, y_{1}) {2} 1".format(i, j, gt_lt[i])) for i, j in label_indices])
        for i, label_class in enumerate(label_classes):
            label_class.scale(0.9).move_to(labels[i].get_center())

        contour_level = Tex(r"Where does $f(x, y) = 1$?")
        contour_level.move_to(UP * 3.5)
        self.play(
            Write(contour_level)
        )
        self.wait()

        self.play(
            *[ReplacementTransform(label, label_class) for label, label_class in zip(labels, label_classes)]
        )
        self.wait()

        self.play(
            dots[0].animate.set_color(OUTSIDE_COLOR),
            dots[1].animate.set_color(INSIDE_COLOR),
            dots[2].animate.set_color(INSIDE_COLOR),
            dots[3].animate.set_color(OUTSIDE_COLOR)
        )
        self.wait()

        top_proporitions = [1/2, 1/4, 3/8, 5/16, 9/32]
        bottom_proportions = [1/2, 3/4, 7/8, 13/16, 27/32, 53/64]

        ordered_corners = self.get_ordered_corners(case_square)
        top_line = Line(ordered_corners[0], ordered_corners[1])
        bottom_line = Line(ordered_corners[2], ordered_corners[3])
        top_dot = Dot().move_to(top_line.point_from_proportion(1/2))
        bottom_dot = Dot().move_to(bottom_line.point_from_proportion(1/2))
        for i in range(len(top_proporitions)):
            if i == 0:
                self.play(
                    GrowFromCenter(top_dot),
                    GrowFromCenter(bottom_dot)
                )
                continue
            self.play(
                top_dot.animate.move_to(top_line.point_from_proportion(top_proporitions[i])),
                bottom_dot.animate.move_to(bottom_line.point_from_proportion(bottom_proportions[i]))
            )
            if i == len(top_proporitions) - 1:
                self.play(
                    top_dot.animate.set_color(CONTOUR_COLOR),
                    Flash(top_dot),
                    bottom_dot.animate.move_to(bottom_line.point_from_proportion(bottom_proportions[-1]))
                )

        self.play(
            bottom_dot.animate.set_color(CONTOUR_COLOR),
            Flash(bottom_dot)
        )
        self.wait()

        contour_line = Line(top_dot.get_center(), bottom_dot.get_center())
        contour_line.set_color(CONTOUR_COLOR)
        self.play(
            Create(contour_line)
        )
        self.wait()

        self.play(
            FadeOut(contour_line),
            FadeOut(top_dot),
            FadeOut(bottom_dot),
        )

        return dots, label_classes, contour_level

    def get_ordered_corners(self, case_square):
        # It's somewhat annoying aesthetically that square 
        # corners go counterclockwise from top right
        order = [1, 0, 2, 3]
        return [case_square.get_vertices()[i] for i in order]

    def motivate_lerp(self, case_square, dots, label_classes):
        ordered_corners = self.get_ordered_corners(case_square)
        top_line = Line(ordered_corners[0], ordered_corners[1])
        bottom_line = Line(ordered_corners[2], ordered_corners[3])

        values = [1.5, 0, 2.5, 0.5]
        label_indices = [(0, 0), (1, 0), (0, 1), (1, 1)]
        label_values = VGroup(*[MathTex("f(x_{0}, y_{1}) = {2}".format(i, j, values[j * 2 + i])) for i, j in label_indices])
        for i, label_value in enumerate(label_values):
            label_value.scale(0.9).move_to(label_classes[i].get_center())

        self.play(
            *[ReplacementTransform(label_class, label_value) for label_class, label_value in zip(label_classes, label_values)]
        )
        self.wait()

        self.play(
            Indicate(label_values[0])
        )

        self.wait()

        self.play(
            Indicate(label_values[1])
        )
        self.wait()

        top_line_highlight = top_line.copy().set_color(CONTOUR_COLOR)

        bottom_line_highlight = bottom_line.copy().set_color(CONTOUR_COLOR)

        self.play(
            ShowPassingFlash(top_line_highlight, time_width=1
                )
        )
        self.wait()



        all_dashed_lines = VGroup()
        dashed_line = DashedLine(UP * 0.5, ORIGIN)
        all_dashed_lines.add(dashed_line.move_to(top_line.point_from_proportion(1/3)))
        all_dashed_lines.add(dashed_line.copy().move_to(top_line.point_from_proportion(2/3)))
        all_dashed_lines.add(dashed_line.copy().move_to(bottom_line.point_from_proportion(1/4)))
        all_dashed_lines.add(dashed_line.copy().move_to(bottom_line.point_from_proportion(2/4)))
        all_dashed_lines.add(dashed_line.copy().move_to(bottom_line.point_from_proportion(3/4)))
        self.play(
            Write(all_dashed_lines[0]),
            Write(all_dashed_lines[1]),
        )
        self.wait()
        top_dot = Dot().set_color(CONTOUR_COLOR).move_to(all_dashed_lines[0].get_center())
        approx_top = MathTex(r"f(x, y) \approx 1").scale(0.7)
        approx_top.next_to(top_dot, DOWN)
        self.play(
            Write(approx_top)
        )
        
        self.play(
            GrowFromCenter(top_dot),
            Flash(top_dot)
        )
        self.wait()

        self.play(
            ShowPassingFlash(bottom_line_highlight, time_width=1)
        )
        self.wait()

        self.play(
            Write(all_dashed_lines[2]),
            Write(all_dashed_lines[3]),
            Write(all_dashed_lines[4]),
        )

        bottom_dot = top_dot.copy().move_to(all_dashed_lines[4].get_center())

        approx_bottom = approx_top.copy().next_to(bottom_dot, UP)
        self.play(
            Write(approx_bottom)
        )
        self.play(
            GrowFromCenter(bottom_dot),
            Flash(bottom_dot)
        )
        self.wait()

        contour_line = Line(top_dot.get_center(), bottom_dot.get_center())
        contour_line.set_color(CONTOUR_COLOR)
        approx_top.add_background_rectangle()
        approx_bottom.add_background_rectangle()
        self.add_foreground_mobjects(approx_top, approx_bottom)
        self.play(
            Create(contour_line)
        )
        self.wait()

        shift_amount = LEFT * 3
        self.play(
            case_square.animate.shift(shift_amount),
            dots.animate.shift(shift_amount),
            label_values[0].animate.shift(shift_amount + UP * SMALL_BUFF / 2).scale(0.8),
            label_values[1].animate.shift(shift_amount + UP * SMALL_BUFF / 2).scale(0.8),
            label_values[2].animate.shift(shift_amount + DOWN * SMALL_BUFF / 2).scale(0.8),
            label_values[3].animate.shift(shift_amount + DOWN * SMALL_BUFF / 2).scale(0.8),
            contour_line.animate.shift(shift_amount),
            all_dashed_lines.animate.shift(shift_amount),
            top_dot.animate.shift(shift_amount),
            bottom_dot.animate.shift(shift_amount),
            approx_top.animate.shift(shift_amount),
            approx_bottom.animate.shift(shift_amount),
        )
        self.wait()

        line_example = Line(LEFT * 2, RIGHT * 2).set_color(BLUE_D)
        left_dot, right_dot = Dot().set_color(INSIDE_COLOR), Dot().set_color(OUTSIDE_COLOR)
        line_example.move_to(RIGHT * 3 + UP * 2)
        left_dot.move_to(line_example.get_start())
        right_dot.move_to(line_example.get_end())

        self.play(
            GrowFromCenter(left_dot),
            GrowFromCenter(right_dot),
            GrowFromCenter(line_example)
        )
        self.wait()

        left_label = MathTex("f(x_0, y_0) = v_0").scale(0.9)
        right_label = MathTex("f(x_1, y_0) = v_1").scale(0.9)
        left_label.next_to(left_dot, DOWN)
        right_label.next_to(right_dot, DOWN)

        self.play(
            Write(left_label),
            Write(right_label)
        )
        self.wait()

        contour_dot = Dot().move_to(line_example.point_from_proportion(1/2))
        contour_dot.set_color(CONTOUR_COLOR)
        target_label = MathTex(r"f(x, y) \approx 1").scale(0.9)
        target_label.next_to(contour_dot, UP)
        self.play(
            GrowFromCenter(contour_dot),
            Write(target_label),
        )
        self.wait()
        
        result_y = MathTex("y = y_0").scale(0.9)
        result_y.move_to(line_example.get_center()).shift(DOWN * 1.5)

        self.play(
            Write(result_y)
        )
        self.wait()

        observe_x = MathTex(r"\frac{x - x_0}{x_1 - x_0}", r" \approx ", r"\frac{f(x, y) - f(x_0, y_0)}{f(x_1, y_0) - f(x_0, y_0)}")
        observe_x.scale(0.9)
        observe_x.next_to(result_y, DOWN)

        self.play(
            FadeIn(observe_x)
        )
        self.wait()

        self.play(
            target_label.animate.set_color(CONTOUR_COLOR),
            observe_x[2][:6].animate.set_color(CONTOUR_COLOR),
        )
        self.wait()

        new_observe_x = MathTex(r"\frac{x - x_0}{x_1 - x_0}", r" \approx ", r" \frac{1 - v_0}{v_1 - v_0}")
        new_observe_x.scale(0.9).move_to(observe_x.get_center())

        self.play(
            ReplacementTransform(observe_x[0], new_observe_x[0]),
            ReplacementTransform(observe_x[1], new_observe_x[1]),
            ReplacementTransform(observe_x[2], new_observe_x[2]),
            target_label.animate.set_color(WHITE),
        )
        self.wait()

        final_result = MathTex(r"x \approx  x_0 + \frac{1 - v_0}{v_1 - v_0}(x_1 - x_0)")
        final_result.scale(0.9)
        final_result.next_to(new_observe_x, DOWN)

        self.play(
            Write(final_result)
        )
        self.wait()

        surround_rect_top = SurroundingRectangle(result_y)
        surround_rect_bottom = SurroundingRectangle(final_result)

        

        linear_interpolation = Tex("Linear Interpolation").set_color(YELLOW)
        linear_interpolation.move_to(RIGHT * 3 + DOWN * 3)

        self.play(
            Create(surround_rect_top),
            Create(surround_rect_bottom),
            Write(linear_interpolation)
        )
        self.wait()


class EnergyFieldsTakeTwoCut(MarchingSquaresUtils):
    def construct(self):
        self.wait()
        SIM_TIME = 11
        colors = [RED, ORANGE, YELLOW, PURE_GREEN, GREEN, BLUE]
        position = LEFT * 4
        radius = 1
        contour_values = [5, 2, 1, 0.5, 0.3, 0.2]
        contour_function_values = [MathTex(f"f(x, y) = {value}") for value in contour_values]
        contour_function_values = VGroup(*contour_function_values).arrange(UP)

        left_ball = Ball(radius=radius, color=BLACK)
        right_ball = Ball(radius=radius, color=BLACK)
        
        velocity = RIGHT * 0.5
        
        left_ball.move_to(position)
        left_ball.set_center(position)
        left_ball.velocity = velocity
        
        right_ball.move_to(-position)
        right_ball.set_center(-position)
        right_ball.velocity = -velocity
        metaballs = VGroup(left_ball)


        left_energy_field = self.get_energy_fields(colors, metaballs)
        
        order = [2, 1, 0, 3, 4, 5]
        for index in order:
            contour_function_values[index].set_color(colors[index])
            self.add(contour_function_values[index], left_energy_field[index])
            self.wait()

        f_xy = contour_function_values[2][0][:6].copy()
        f_xy.next_to(left_energy_field, DOWN)
        self.play(
            TransformFromCopy(contour_function_values[2][0][:6], f_xy),
            FadeOut(contour_function_values)
        )
        self.wait()

        metaballs = VGroup(right_ball)
        right_energy_field = self.get_energy_fields(colors, metaballs)
        contour_g_function_values = [MathTex(f"g(x, y) = {value}") for value in contour_values]
        contour_g_function_values = VGroup(*contour_g_function_values).arrange(UP)

        for index in order:
            contour_g_function_values[index].set_color(colors[index])
            self.add(contour_g_function_values[index], right_energy_field[index])
            self.wait()

        g_xy = contour_g_function_values[2][0][:6].copy()
        g_xy.next_to(right_energy_field, DOWN)
        self.play(
            TransformFromCopy(contour_g_function_values[2][0][:6], g_xy),
            FadeOut(contour_g_function_values)
        )
        self.wait()

        metaballs = VGroup(left_ball, right_ball)
        for index in order:
            self.remove(left_energy_field[index])
            self.remove(right_energy_field[index])

        dual_energy_fields = self.get_energy_fields(colors, metaballs)
        self.add_foreground_mobject(dual_energy_fields)
        
        def update_metaballs(metaballs, dt):
            for ball in metaballs:
                new_position = ball.get_center() + ball.velocity * dt
                ball.shift(ball.velocity * dt)
                ball.set_center(new_position)

            new_dual_energy_fields = self.get_energy_fields(colors, metaballs)
            for i, energy_field in enumerate(dual_energy_fields):
                dual_energy_fields[i].become(new_dual_energy_fields[i])

        metaballs.add_updater(update_metaballs)
        self.add(metaballs)
        self.wait(SIM_TIME)
        metaballs.clear_updaters()

        surround_rect = SurroundingRectangle(VGroup(dual_energy_fields, f_xy, g_xy), buff=MED_SMALL_BUFF)
        bigger_rect = Rectangle(height=surround_rect.height + 0.2, width=surround_rect.width + 2.5)
        bigger_rect.set_stroke(color=YELLOW, width=6)

        sum_implicit_functions = MathTex("h(x, y) = f(x, y) + g(x, y)")
        sum_implicit_functions[0][7:13].set_color(YELLOW)
        sum_implicit_functions[0][14:].set_color(YELLOW)
        sum_implicit_functions.next_to(bigger_rect, UP)

        self.play(
            Write(bigger_rect)
        )
        self.play(
            Write(sum_implicit_functions)
        )
        self.wait()

        self.show_metaball_implicit_functions(f_xy, g_xy, bigger_rect, sum_implicit_functions, metaballs, dual_energy_fields, sum_implicit_functions)

        metaballs.add_updater(update_metaballs)
        self.add(metaballs)
        self.wait(4)
        self.play(
            FadeOut(sum_implicit_functions),
            FadeOut(bigger_rect),
            FadeOut(f_xy),
            FadeOut(g_xy),
        )
        metaballs.clear_updaters()


        # switch directions
        metaballs[0].velocity = -metaballs[0].velocity
        metaballs[1].velocity = -metaballs[1].velocity

        proton = get_proton()

        vector_field = self.get_vector_field(metaballs[1].get_center(), metaballs[0].get_center(), velocity)
        left_proton = get_proton().move_to(metaballs[1].get_center())
        right_proton = get_proton().move_to(metaballs[0].get_center())
        self.play(
            GrowFromCenter(left_proton),
            GrowFromCenter(right_proton)
        )
        self.play(
            LaggedStartMap(Create, vector_field),
            run_time=2
        )
        self.wait()
        
        def second_updater_with_vector_field(metaballs, dt):
            for ball in metaballs:
                new_position = ball.get_center() + ball.velocity * dt
                ball.shift(ball.velocity * dt)
                ball.set_center(new_position)
            left_proton.shift(metaballs[1].velocity * dt)
            right_proton.shift(metaballs[0].velocity * dt)

            new_dual_energy_fields = self.get_energy_fields(colors, metaballs)
            for i, energy_field in enumerate(dual_energy_fields):
                dual_energy_fields[i].become(new_dual_energy_fields[i])

            new_vector_field = self.get_vector_field(metaballs[1].get_center(), metaballs[0].get_center(), velocity)
            vector_field.become(new_vector_field)

        metaballs.add_updater(second_updater_with_vector_field)

        self.add_foreground_mobject(vector_field)
        self.add_foreground_mobjects(left_proton, right_proton)
        self.add_foreground_mobject(dual_energy_fields)

        self.add(metaballs)
        self.wait(3)
        metaballs.clear_updaters()
        self.remove(metaballs)

        self.show_electric_potential_connection(left_proton, right_proton, dual_energy_fields, vector_field)

    def show_metaball_implicit_functions(self, f_xy, g_xy, bigger_rect, sum_implicit_functions, metaballs, dual_energy_fields, h_xy):
        original_fxy_pos = f_xy.get_center()
        original_gxy_pos = g_xy.get_center()

        right_ball, left_ball = metaballs
        left_dot = Dot().move_to(left_ball.get_center())
        right_dot = Dot().move_to(right_ball.get_center())
        self.remove(metaballs)
        self.bring_to_back(dual_energy_fields)

        self.play(
            FadeOut(bigger_rect),
            f_xy.animate.to_edge(LEFT).shift(DOWN * 0.5),
            g_xy.animate.to_edge(RIGHT * 4).shift(DOWN * 0.5)
        )
        self.wait()

        x1_y1 = MathTex("(x_1, y_1)").scale(0.8).add_background_rectangle()
        x1_y1.add_background_rectangle()
        x1_y1.next_to(left_dot, UP)

        x2_y2 = MathTex("(x_2, y_2)").scale(0.8).add_background_rectangle()
        x2_y2.add_background_rectangle()
        x2_y2.next_to(right_dot, UP)

        self.play(
            GrowFromCenter(left_dot),
            FadeIn(x1_y1),
            GrowFromCenter(right_dot),
            FadeIn(x2_y2)
        )
        self.wait()

        potential_point = RIGHT * 2 + DOWN * 2

        contour_dot = Dot().move_to(potential_point).set_color(CONTOUR_COLOR)
        x_y = MathTex("(x, y)").scale(0.8).set_color(CONTOUR_COLOR)
        x_y.next_to(contour_dot, DOWN).shift(UP * SMALL_BUFF * 2)
        self.play(
            GrowFromCenter(contour_dot),
            Write(x_y)
        )
        self.wait()

        r_1_line = DashedLine(left_dot.get_center(), potential_point)
        r_2_line = DashedLine(right_dot.get_center(), potential_point)

        self.play(
            Write(r_1_line),
            Write(r_2_line),
        )

        r_1 = MathTex("r_1").scale(0.8).add_background_rectangle()
        r_2 = MathTex("r_2").scale(0.8).add_background_rectangle()

        r_1.move_to(r_1_line.get_center())
        r_2.move_to(r_2_line.get_center())

        r_1.shift(LEFT * SMALL_BUFF * 4)
        r_2.shift(RIGHT * SMALL_BUFF * 3)

        self.play(
            FadeIn(r_1),
            FadeIn(r_2),
        )
        self.wait()

        f_result = MathTex(r"= \frac{1}{r_1}").next_to(f_xy, RIGHT, buff=SMALL_BUFF)
        g_result = MathTex(r"= \frac{1}{r_2}").next_to(g_xy, RIGHT, buff=SMALL_BUFF)

        self.play(
            Write(f_result),
            Write(g_result)
        )
        self.wait()

        f_result_transformed = MathTex(r"= \frac{1}{\sqrt{(x - x_1) ^ 2 + (y - y_1) ^ 2}}").scale(0.8)
        g_result_transformed = MathTex(r"= \frac{1}{\sqrt{(x - x_2) ^ 2 + (y - y_2) ^ 2}}").scale(0.8)

        f_result_transformed.next_to(f_xy, RIGHT, buff=SMALL_BUFF * 2)

        g_xy_copy = g_xy.copy().next_to(f_result_transformed, RIGHT * 3)
        g_result_transformed.next_to(g_xy_copy, RIGHT, buff=SMALL_BUFF * 2)
        
        self.play(
            Transform(f_result, f_result_transformed),
            Transform(g_xy, g_xy_copy),
            Transform(g_result, g_result_transformed),
            run_time=2
        )
        self.wait()

        underline = Underline(h_xy).set_color(YELLOW)

        self.play(
            ShowPassingFlash(underline, time_width=2)
        )
        self.wait()

        self.remove(r_1, r_2, r_1_line, r_2_line, f_result, g_result, left_dot, right_dot, x1_y1, x2_y2, x_y, contour_dot)
        self.add(bigger_rect)
        f_xy.move_to(original_fxy_pos)
        g_xy.move_to(original_gxy_pos)
        self.add_foreground_mobject(dual_energy_fields)
        self.wait()

    def show_electric_potential_connection(self, left_proton, right_proton, dual_energy_fields, vector_field):
        self.bring_to_back(dual_energy_fields, vector_field)
        self.wait()
        
        electric_potential_title = Tex("Electric Potential Lines in An Electric Field")
        electric_potential_title.move_to(UP * 3).add_background_rectangle()
        self.play(
            FadeIn(electric_potential_title)
        )
        # self.add_foreground_mobject(electric_potential_title)
        self.wait()

        # contour = dual_energy_fields[-1]
        # potential_point = self.sample_point_from_contour(contour)
        # if potential_point[1] > 0:
        #     potential_point[1] = -potential_point[1]

        potential_point = RIGHT * 1.5 + DOWN * 2.35
        dot = Dot().move_to(potential_point)
        label = Tex(r"Potential $V_P = V_{Q_1} + V_{Q_2}$").scale(0.8)
        label.add_background_rectangle()
        r_1_line = DashedLine(left_proton.get_center(), potential_point)
        r_2_line = DashedLine(right_proton.get_center(), potential_point)

        P_label = MathTex("P").scale(0.8).move_to(potential_point)
        P_label.add_background_rectangle()
        P_label.shift(-normalize(potential_point) * SMALL_BUFF * 5)
        label.move_to(potential_point).shift(normalize(potential_point) * SMALL_BUFF * 5)

        q_1_label = MathTex("Q_1").scale(0.8).add_background_rectangle()
        q_1_label.next_to(left_proton, UP)

        q_2_label = MathTex("Q_2").scale(0.8).add_background_rectangle()
        q_2_label.next_to(right_proton, UP)

        self.play(
            FadeIn(q_1_label),
            FadeIn(q_2_label),
        )
        self.wait()

        self.play(
            GrowFromCenter(dot),
            FadeIn(P_label)
        )
        self.wait()
        self.play(
            FadeIn(label)
        )

        self.play(
            Write(r_1_line),
            Write(r_2_line),
        )

        r_1 = MathTex("r_1").scale(0.8).add_background_rectangle()
        r_2 = MathTex("r_2").scale(0.8).add_background_rectangle()

        r_1.move_to(r_1_line.get_center())
        r_2.move_to(r_2_line.get_center())

        r_1.shift(LEFT * SMALL_BUFF * 4)
        r_2.shift(RIGHT * SMALL_BUFF * 3)

        self.play(
            FadeIn(r_1),
            FadeIn(r_2)
        )
        self.wait()

        V_Q1 = MathTex(r"V_{Q_1} \propto \frac{1}{r_1}").add_background_rectangle()
        V_Q2 = MathTex(r"V_{Q_2} \propto \frac{1}{r_2}").add_background_rectangle()

        V_Q1.to_edge(LEFT).shift(DOWN * 3)
        V_Q2.to_edge(RIGHT).shift(DOWN * 3)

        self.play(
            FadeIn(V_Q1),
            FadeIn(V_Q2)
        )
        self.wait()


    def sample_point_from_contour(self, contour):
        index = np.random.randint(len(contour))
        set_of_lines = contour[index]
        nested_index = np.random.choice(list(range(len(set_of_lines))))
        return set_of_lines[nested_index].get_start()

    def get_energy_fields(self, colors, metaballs):
        dual_energy_fields = VGroup()
        x_step, y_step = 0.125, 0.125
        self.set_sample_space(x_step=x_step, y_step=y_step)
        
        implicit_function = implicit_function_group_of_circles(metaballs)

        contour_values = [5, 2, 1, 0.5, 0.3, 0.2]
        for i, value in enumerate(contour_values):
            condition = get_condition_func(value)
            self.get_implicit_function_samples(implicit_function)
            contour = self.march_squares(condition, implicit_function, value=value)
            contour.set_color(colors[i])
            dual_energy_fields.add(contour)

        return dual_energy_fields

    def get_vector_field(self, left_center, right_center, velocity):
        right_shift = velocity * 1 / self.camera.frame_rate
        left_shift = -velocity * 1 / self.camera.frame_rate
        vector_field = ArrowVectorField(field_function_two_charges(left_center + left_shift, right_center + right_shift))
        return vector_field


class MarchingCubes3DCameraRotation(MarchingCubesUtils):
    def construct(self):
        x_range, y_range, z_range = (-2.5, 2.5), (-2.5, 2.5), (-2.5, 2.5)
        x_step, y_step, z_step = 0.25, 0.25, 0.25 
        self.set_sample_space(
            x_range=x_range, y_range=y_range, z_range=z_range,
            x_step=x_step, y_step=y_step, z_step=z_step
        )
        function_map = self.get_implicit_function_samples(metaball_implicit_function)
        condition = lambda x: x >= 1
        
        # dots = VGroup()

        # color_map = {-1.5: RED, -0.5: GREEN, 0.5: BLUE, 1.5: PURPLE}

        # for point in self.get_sample_space():
        #     print(point)
        #     dot = Dot3D(point=point, color=color_map[point[2]])
        #     dots.add(dot)

        self.set_camera_orientation(phi=75*DEGREES, theta=-60*DEGREES, distance=5)
        ax = ThreeDAxes(x_range=(-6, 6, 1), y_range=(- 5, 5, 1), z_range=(- 4, 4, 1))
        ax.set_stroke(opacity=0.5)
        self.add(ax)
        # self.add(dots)

        mesh = self.march_cubes(condition, implicit_function_sphere)
        self.add(mesh)
        # mesh.set_color_by_gradient(YELLOW, PURE_GREEN, BLUE)
        # self.add(mesh)
        # self.play(
        #     FadeIn(dots)
        # )
        self.begin_ambient_camera_rotation(rate=-0.02)
        self.wait(20)


class Conclusion(Scene):
    def construct(self):
        title = Title("Two Different Learning Paths")

        first_encounter = Tex("My first encounter with marching squares/cubes")
        first_encounter.next_to(title, DOWN * 2)

        self.play(
            Write(title)
        )
        self.wait()
        self.play(
            FadeIn(first_encounter)
        )
        self.wait()

        lecture_component = self.make_component("Lecture")
        examples_component = self.make_component("Examples", color=ORANGE)
        homework_component  = self.make_component("Homework", color=RED)
        exam_component = self.make_component("Exam", color=PURE_GREEN)
        forget_component = self.make_component("Forget", color=BLUE)

        first_encounter_components = VGroup(
            lecture_component, 
            examples_component,
            homework_component,
            exam_component,
            forget_component 
        ).arrange(RIGHT * 2)

        first_encounter_components.next_to(first_encounter, DOWN)

        for component in first_encounter_components:
            self.play(
                Create(component[0]),
                Write(component[1])
            )
            self.wait()

        this_video = Tex("My experience making this video")

        this_video.next_to(first_encounter_components, DOWN * 2)

        self.play(
            FadeIn(this_video)
        )
        self.wait()

        build_component = self.make_component("Build" + "\\\\" + " metaballs")
        marching_squares_component = self.make_component("Marching" + "\\\\" + "squares/cubes", color=ORANGE)
        implement_component = self.make_component("Implement" + "\\\\" + "algorithms", color=RED)
        play_around = self.make_component("Experiment", color=PURE_GREEN)
        remember = self.make_component("Remember" + "\\\\" + "forever?", color=BLUE)

        video_experience_components = VGroup(
            build_component,
            marching_squares_component,
            implement_component,
            play_around,
            remember
        ).arrange(RIGHT * 2)

        video_experience_components.next_to(this_video, DOWN)

        for component in video_experience_components:
            self.play(
                Create(component[0]),
                Write(component[1])
            )
            self.wait()

        brace = Brace(video_experience_components, direction=DOWN)
        self.play(
            GrowFromCenter(brace)
        )
        self.wait()
        hard = Tex("More time-consuming and challenging")
        rewarding = Tex("But also more rewarding")

        hard.next_to(brace, DOWN)
        self.play(
            Write(hard)
        )
        self.wait()
        rewarding.move_to(hard.get_center())

        self.play(
            ReplacementTransform(hard, rewarding)
        )
        self.wait()

    def make_component(self, text, color=YELLOW, scale=0.7):
        # geometry is first index, TextMob is second index
        text_mob = Tex(text).scale(scale)
        rect = Rectangle(color=color, height=1.1, width=2.3)
        return VGroup(rect, text_mob)


class Test(Scene):
    def construct(self):
        triangle = Polygon(LEFT, RIGHT, UP * 2)
        triangle.set_stroke(width=0)
        triangle.set_fill(color=[RED, YELLOW, BLUE], opacity=1)
        triangle.set_sheen_direction([-1, 0, 0])
        # triangle.set_sheen_direction([0, 1, 0]).rotate_sheen_direction(PI / 2)
        self.add(triangle)
        self.wait()

def get_random_point_in_frame(buff=0):
    x_min, x_max = floor(-config.frame_width / 2) + buff, floor(config.frame_width / 2) - buff
    y_min, y_max = floor(-config.frame_height / 2) + buff, floor(config.frame_height / 2) - buff
    return np.array([np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max), 0])

def implicit_function_circle(sample_point, circle):
    """
    Implicit function of circle where
    point inside circle returns >= 1
    and point outside returns <= 1
    @param: sample_point - [x, y, 0]
    @circle: Geometry.Circle object
    """
    x, y, _ = sample_point
    c_x, c_y, _ = circle.get_center()
    return circle.radius ** 2 / ((x - c_x) ** 2 + (y - c_y) ** 2)

def implicit_function_circle_sqrt(sample_point, circle):
    """
    Alternate implicit function of circle where
    point inside circle returns >= 1
    and point outside returns <= 1
    Uses sqrt instead of squared distance
    @param: sample_point - [x, y, 0]
    @circle: Geometry.Circle object
    """
    x, y, _ = sample_point
    c_x, c_y, _ = circle.get_center()
    return circle.radius / np.sqrt(((x - c_x) ** 2 + (y - c_y) ** 2))

def metaball_higher_order_function(ball):
    """
    Higher order function variant of above function
    """
    def func(sample_point):
        x, y, _ = sample_point
        c_x, c_y, _ = ball.get_center()
        return ball.radius ** 2 / ((x - c_x) ** 2 + (y - c_y) ** 2)
    return func

def test_function(sample_point):
    x, y, _ = sample_point
    # circle.get_center()
    return max(1 / ((x - 1.2) ** 2 + y ** 2),  1 / ((x + 1) ** 2 + y ** 2))

def batman_function(sample_point):
    x, y, _ = sample_point
    from scipy import sqrt
    eq1 = ((x/7)**2*sqrt(abs(abs(x)-3)/(abs(x)-3))+(y/3)**2*sqrt(abs(y+3/7*sqrt(33))/(y+3/7*sqrt(33)))-1)
    eq2 = (abs(x/2)-((3*sqrt(33)-7)/112)*x**2-3+sqrt(1-(abs(abs(x)-2)-1)**2)-y)
    eq3 = (9*sqrt(abs((abs(x)-1)*(abs(x)-.75))/((1-abs(x))*(abs(x)-.75)))-8*abs(x)-y)
    eq4 = (3*abs(x)+.75*sqrt(abs((abs(x)-.75)*(abs(x)-.5))/((.75-abs(x))*(abs(x)-.5)))-y)
    eq5 = (2.25*sqrt(abs((x-.5)*(x+.5))/((.5-x)*(.5+x)))-y)
    eq6 = (6*sqrt(10)/7+(1.5-.5*abs(x))*sqrt(abs(abs(x)-1)/(abs(x)-1))-(6*sqrt(10)/14)*sqrt(4-(abs(x)-1)**2)-y)
    # transformation so contour level is 1
    return eq1 * eq2 * eq3 * eq4 * eq5 * eq6 + 1 

def heart_function(sample_point):
    x, y, _ = sample_point
    from scipy import sqrt
    return (x ** 2 + (y - sqrt(abs(x))) ** 2)

def sin_function(sample_point):
    x, y, _ = sample_point
    from numpy import sin
    from numpy import cos
    return sin(sin(x)) - cos(sin(x * y)) - cos(x) + 1

def get_condition_func(value):
    return lambda x: x >= value

def implicit_function_group_of_circles(circles):
    def func(sample_point):
        # return test_function(sample_point, circles[0])
        # return sum([test_function(sample_point, circle) for circle in circles])
        return sum([implicit_function_circle(sample_point, circle) for circle in circles])
    return func

def implicit_function_group_of_circles_sqrt(circles):
    def func(sample_point):
        # return test_function(sample_point, circles[0])
        # return sum([test_function(sample_point, circle) for circle in circles])
        return sum([implicit_function_circle_sqrt(sample_point, circle) for circle in circles])
    return func

def get_func_val_from_map(function_map, key, implicit_function):
    if key in function_map:
        return function_map[key]
    return implicit_function(np.array([key[0], key[1], 0]))

def get_func_val_from_map_3d(function_map, key, implicit_function):
    if key in function_map:
        return function_map[key]
    return implicit_function(np.array(key))


def implicit_function_sphere(sample_point):
    return 1 / np.linalg.norm(sample_point)

def metaball_implicit_function(sample_point):
    sphere_c1 = np.array([1, 0, 0])
    sphere_c2 = np.array([-1, 0, 0])
    x, y, z = sample_point
    c_x1, c_y1, c_z1 = sphere_c1
    c_x2, c_y2, c_z2 = sphere_c2
    first_contribution = 1 / ((x - c_x1) ** 2 + (y - c_y1) ** 2 + (z - c_z1) ** 2)
    second_contribution = 1 / ((x - c_x2) ** 2 + (y - c_y2) ** 2 + (z - c_z2) ** 2)
    return first_contribution + second_contribution

def two_metaball_implicit_function(c1, c2):
    def implicit_function(sample_point):
        x, y, z = sample_point
        c_x1, c_y1, c_z1 = c1
        c_x2, c_y2, c_z2 = c2
        first_contribution = 1 / ((x - c_x1) ** 2 + (y - c_y1) ** 2 + (z - c_z1) ** 2)
        second_contribution = 1 / ((x - c_x2) ** 2 + (y - c_y2) ** 2 + (z - c_z2) ** 2)
        return first_contribution + second_contribution
    return implicit_function

def get_computer_mob():
    outer_rect = RoundedRectangle(corner_radius=SMALL_BUFF)
    inner_rect = RoundedRectangle(corner_radius=SMALL_BUFF).scale(0.8)

    inner_rect.set_color(GRAY_A)
    inner_rect.set_fill(color=GRAY_A, opacity=1)
    outer_rect.set_color(GRAY_D)
    outer_rect.set_fill(color=GRAY_D, opacity=1)

    base = Rectangle(height=outer_rect.get_height() - inner_rect.get_height(), width=outer_rect.get_width() * 0.7)
    base.set_color(GRAY_D).set_fill(color=GRAY_D, opacity=1)
    connector = Rectangle(height=base.get_width() * 0.3 * 0.8, width=base.get_width() * 0.3)
    connector.set_color(GRAY_D).set_fill(color=GRAY_D, opacity=1)

    connector.next_to(outer_rect, DOWN, buff=0)
    base.next_to(connector, DOWN, buff=0)

    computer_mob = VGroup(outer_rect, inner_rect, connector, base)

    return computer_mob

def field_function_with_center(center):
    def field_function(pos):
        if np.isclose(np.linalg.norm(pos - center), 0):
            return 0 * RIGHT
        return (1 / np.linalg.norm(pos - center)) * normalize(pos - center)
    return field_function

def field_function_two_charges(c1, c2):
    def field_function(pos):
        if np.isclose(np.linalg.norm(pos - c1), 0):
            return 0 * RIGHT
        if np.isclose(np.linalg.norm(pos - c2), 0):
            return 0 * RIGHT
        func1 = field_function_with_center(c1)
        func2 = field_function_with_center(c2)
        return func1(pos) + func2(pos)
    return field_function

def get_proton():
    circle = Circle(radius=0.3).set_stroke(color=WHITE, width=1)
    circle.set_fill(color=PURE_RED, opacity=1)
    plus_vert = Rectangle(height=0.25, width=SMALL_BUFF/8).set_color(WHITE)
    plus_vert.set_fill(color=WHITE, opacity=1)
    plus_horiz = Rectangle(height=SMALL_BUFF/8, width=0.25).set_color(WHITE)
    plus_horiz.set_fill(color=WHITE, opacity=1)
    plus = VGroup(plus_vert, plus_horiz)
    return VGroup(circle, plus)

def get_normal_towards_origin(a, b, away=False):
    x1, y1 = a[0], a[1]
    x2, y2 = b[0], b[1]
    dx = x2 - x1
    dy = y2 - y1
    n1 = np.array([-dy, dx, 0])
    n2 = np.array([dy, -dx, 0])
    AO = ORIGIN - a
    if np.dot(AO, n1) > 0:
        if away:
            return n2
        return n1
    if away:
        return n1
    return n2

def position_to_color(position):
    """
    Interpolates color based on position for 
    visually appealing colors in marching squares
    """
    SKY_BLUE = "#1919FF"
    MONOKAI_GREEN = "#A6E22E"
    GREEN_SCREEN = "#00FF00"
    colors = [MONOKAI_GREEN, GREEN_SCREEN, BLUE, SKY_BLUE]
    all_positions = [LEFT * 7.1, LEFT * 2.4, RIGHT * 2.4, RIGHT * 7.1]

    closest_positions = sorted(all_positions, key=lambda x: np.linalg.norm(position - x))[:2]
    closest_positions = [position.tolist() for position in closest_positions]
    all_positions_list = [p.tolist() for p in all_positions]
    a, b = closest_positions
    a_index = all_positions_list.index(a)
    b_index = all_positions_list.index(b)
    alpha = (position[0] - a[0]) / (b[0] - a[0])
    if alpha < 0:
        alpha = 0
    if alpha > 1:
        alpha = 1
    return interpolate_color(colors[a_index], colors[b_index], alpha)


def check_point_inside(point, polygon):
    """
    @param point: test point (type np.array[x, y, z])
    @param polygon: convex polygon (type Geometry.Polygon)
    return: True if point is inside polygon, False otherwise
    """
    vertices = polygon.get_vertices()
    u_indices = list(range(len(vertices)))
    v_indices = [len(vertices) - 1] + list(range(len(vertices) - 1))
    result = False
    for i, j in zip(u_indices, v_indices):
        condition1 = (vertices[i][1] > point[1]) != (vertices[j][1] > point[1]) 
        condition2 = (point[0] < (vertices[j][0] - vertices[i][0]) * 
            (point[1] - vertices[i][1]) / (vertices[j][1] - vertices[i][1]) + vertices[i][0])
        if condition1 and condition2:
            result = not result

    return result
