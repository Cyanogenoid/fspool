import random
import numpy as np

from manimlib.imports import *


LB = "\\left\\{%}\n\\space "
RB = "%{\n\\right\\}"


class Outro(Scene):
    def construct(self):
        self.left_side = VGroup(
            TextMobject("Set encoder").shift(UP),
            TextMobject("Bottleneck problem", color=RED),
            TextMobject(r"\textbf{FSPool}").shift(1.2 * DOWN).set_color_by_gradient(BLUE, GREEN),
        )
        self.right_side = VGroup(
            TextMobject("Set decoder").shift(UP),
            TextMobject("Responsibility problem", color=RED),
            TextMobject(r"\textbf{FSUnpool}").shift(1.2 * DOWN).set_color_by_gradient(BLUE, GREEN),
        )

        self.left_side.shift(3 * LEFT)
        self.right_side.shift(3 * RIGHT)

        self.title_line1 = TextMobject(r"\textbf{FSPool}: Learning Set Representations", stroke_color=WHITE)\
            .scale(1.3)\
            .set_color_by_gradient(BLUE, GREEN)
        self.title_line2 = TextMobject("with Featurewise Sort Pooling", stroke_color=WHITE)\
            .scale(1.3)\
            .next_to(self.title_line1, DOWN)\
            .set_color_by_gradient(BLUE, GREEN)

        link = "https://arxiv.org/abs/1906.02795"
        self.subtitle = TextMobject(f"Paper: {link}", tex_to_color_map={link: BLUE})\
            .next_to(self.title_line2, DOWN)
        link = "https://github.com/Cyanogenoid/fspool"
        self.subtitle2 = TextMobject(f"Code: {link}", tex_to_color_map={link: GREEN})\
            .next_to(self.subtitle, DOWN)
        self.title_line1.shift(UP)
        self.title_line2.shift(UP)

        self.show_summary()
        self.end_screen()
    
    def end_screen(self):
        self.play(Transform(
            VGroup(self.left_side, self.right_side),
            VGroup(
                self.title_line1,
                self.title_line2,
                self.subtitle,
                self.subtitle2,
            ),
        ))

    def show_summary(self):
        for thing in self.left_side:
            self.play(Write(thing), run_time=1)
        for thing in self.right_side:
            self.play(Write(thing), run_time=1)


class FSUnpool(Scene):
    def construct(self):
        self.ae = Autoencoder()
        self.p = TexMobject(r"\mathbf{P}")
        self.fspool = TextMobject("FSPool")
        self.fsunpool = TextMobject("FSUnpool")
        self.dspn = TextMobject("Deep Set Prediction Networks")
        self.dspn_link = TextMobject("https://arxiv.org/abs/1906.06565")

        self.p.next_to(self.ae.encoder, UP)
        self.fspool.scale(0.4).move_to(self.ae.encoder)
        self.fsunpool.scale(0.4).move_to(self.ae.decoder)

        VGroup(self.dspn, self.dspn_link).arrange(DOWN).next_to(self.ae.latent, UP, buff=1.5*LARGE_BUFF)
    
        self.fade_in()
        self.take_out_permutation()
        self.move_and_insert_to_decoder()
        self.show_rotate()

    def show_rotate(self):
        self.play(
            ApplyMethod(Group(self.ae.input_set).rotate, -1.5 * 360 * DEGREES, path_arc=-1.5 * 360 * DEGREES),
            ApplyMethod(Group(self.ae.output_set).rotate, -1.5 * 360 * DEGREES, path_arc=-1.5 * 360 * DEGREES),
            run_time=6,
            rate_func=linear,
        )
        self.play(
            Write(self.dspn),
            Write(self.dspn_link),
            run_time=1.2,
        )
        self.play(
            FadeOut(self.ae),
            FadeOut(self.fspool),
            FadeOut(self.fsunpool),
            FadeOut(self.dspn),
            FadeOut(self.dspn_link),
        )
    
    def move_and_insert_to_decoder(self):
        self.play(self.p.next_to, self.ae.decoder, UP)
        self.play(
            FadeOutAndShiftDown(self.p),
            Write(self.fsunpool),
        )

    def fade_in(self):
        self.play(
            FadeIn(self.ae),
            FadeIn(self.fspool),
        )
    
    def take_out_permutation(self):
        inv_p = TexMobject("\mathbf{P}^{-1}").move_to(self.p, aligned_edge=DL)
        p_transpose = TexMobject("\mathbf{P}^T").move_to(self.p, aligned_edge=DL)

        self.play(FadeInFromDown(self.p))
        self.play(Transform(self.p, inv_p))
        self.play(Transform(self.p, p_transpose))


class SectionOverview(Scene):
    def construct(self):
        self.ae = Autoencoder()
        for dot in self.ae.output_set:
            dot.set_color(WHITE)

        self.fade_in()
        self.zoom()
    
    def fade_in(self):
        things_to_fade_in = [
            self.ae.input_set,
            self.ae.encoder,
            self.ae.encoder_text,
            self.ae.arrow0,
            self.ae.latent,
            self.ae.arrow1,
        ]
        self.play(*[
            FadeIn(x) for x in things_to_fade_in
        ])
        things_to_fade_in = [
            self.ae.arrow2,
            self.ae.decoder,
            self.ae.decoder_text,
            self.ae.output_set,
            self.ae.arrow3,
        ]
        self.play(*[
            FadeIn(x) for x in things_to_fade_in
        ])

    def zoom(self):
        shift = -self.ae.encoder.get_center()
        all_objects = self.ae

        target = all_objects.copy()
        target.shift(shift)
        target.scale(15, about_point=ORIGIN)

        self.play(Transform(all_objects, target), rate_func=smooth, run_time=1.5)
        
    def blink_encoder(self):
        self.play(
            Indicate(self.ae.encoder_text),
            Indicate(self.ae.encoder),
        )
    
    def blink_decoder(self):
        self.play(
            Indicate(self.ae.decoder_text),
            Indicate(self.ae.decoder),
        )


class DecoderBasics(Scene):
    def construct(self):
        self.ae_original = Autoencoder()
        self.output_set_original = self.ae_original.output_set.copy()
        for dot in self.ae_original.output_set:
            dot.set_color(WHITE)
        shift = -self.ae_original.encoder.get_center()

        self.ae = Autoencoder()
        self.ae.shift(shift)
        self.ae.scale(15, about_point=ORIGIN)
        self.add(self.ae)

        self.rnn = TextMobject("RNN").scale(0.6).move_to(self.ae_original.decoder)
        self.mlp = TextMobject("MLP").scale(0.6).move_to(self.ae_original.decoder)

        self.zoom_out()
        self.write_mlp()
        self.show_outputs()
    
    def zoom_out(self):
        self.play(Transform(self.ae, self.ae_original))

    def write_mlp(self):
        self.play(Write(self.rnn))
        self.play(Transform(self.rnn, self.mlp))

    def show_outputs(self):
        outputs = [
            ["x_1"],
            ["y_1"],
            ["x_2"],
            ["y_2"],
            ["x_3"],
            ["y_3"],
            ["x_4"],
            ["y_4"],
        ]
        matrix = Matrix(outputs).move_to(self.ae.output_set)
        coloured_matrix = Matrix(outputs).move_to(self.ae.output_set)
        entries = coloured_matrix.get_entries()
        for i, color in enumerate([RED_D, PURPLE_A, GOLD_B, BLUE_E]):
            entries[2 * i + 0].set_color(color)
            entries[2 * i + 1].set_color(color)
        
        target = VGroup(
            *matrix.get_entries()[0:4],
            *matrix.get_entries()[6:8],
            *matrix.get_entries()[4:6],
        )
        self.play(
            Write(matrix.brackets),
            Transform(self.ae.output_set, target),
        )
        
        target = VGroup(
            *coloured_matrix.get_entries()[0:4],
            *coloured_matrix.get_entries()[6:8],
            *coloured_matrix.get_entries()[4:6],
        )
        self.play(Transform(self.ae.output_set, target))

        self.play(
            FadeOut(matrix.brackets),
            Transform(self.ae.output_set, self.output_set_original),
            FadeOut(self.rnn),
        )


class EncoderBasics(Scene):
    def construct(self):
        cmap = {
            r"\mathbf{a}": BLUE,
            r"\mathbf{b}": YELLOW,
            r"\mathbf{c}": GREEN,
        }
        self.set =  TexMobject(LB + r"\mathbf{a}, \mathbf{b}, \mathbf{c}" + RB, tex_to_color_map=cmap)
        self.set2 = TexMobject(LB + r"\mathbf{b}, \mathbf{a}, \mathbf{c}" + RB, tex_to_color_map=cmap)
        self.transform_element = TextMobject("neural net").scale(0.8)
        self.transformed =  TexMobject(LB + r"f(\mathbf{a}), f(\mathbf{b}), f(\mathbf{c})" + RB, tex_to_color_map=cmap)
        self.transformed2 = TexMobject(LB + r"f(\mathbf{b}), f(\mathbf{a}), f(\mathbf{c})" + RB, tex_to_color_map=cmap)
        self.sum = TexMobject(r"\sum", tex_to_color_map=cmap).scale(0.9)
        self.result = TexMobject(r"\mathbf{y}")
        self.result0 = TexMobject("0")
        self.transform_element2 = TextMobject("neural net").scale(0.8)
        self.end = TexMobject(r"g(\mathbf{y})")

        cmap = {
            "0": YELLOW,
            "1": BLUE,
            "2": PURPLE,
            "3": RED,
            "4": GREEN,
        }
        self.example0 = TexMobject(LB, "-2,", "0,", "1,", "1", RB, tex_to_color_map=cmap)
        self.example1 = TexMobject(LB, "-3,",  "1,", "1,", "1", RB, tex_to_color_map=cmap)
        self.example2 = TexMobject(LB,  "0,",  "0,", "0,", "0", RB, tex_to_color_map=cmap)
        self.example3 = TexMobject(LB, "-3,", "-2,", "1,", "4", RB, tex_to_color_map=cmap)

        arrow_args = {
            'color': BLUE,
            'stroke_width': 4,
            'tip_length': DEFAULT_ARROW_TIP_LENGTH / 2,
            'buff': MED_LARGE_BUFF,
        }
        self.arrows = [Arrow(ORIGIN, 1.5 * DOWN, color=GREY) for _ in range(3)]

        self.objects = VGroup(
            self.set,
            self.arrows[0],
            self.transformed,
            self.arrows[1],
            self.result,
            self.arrows[2],
            self.end,
        ).arrange(DOWN, buff=MED_SMALL_BUFF)

        self.example0.shift(LEFT * 1.5).shift(UP)
        self.example1.shift(LEFT * 1.5)
        self.example2.shift(1.5 * RIGHT).shift(UP)
        self.example3.shift(1.5 * RIGHT)
        VGroup(self.example0, self.example1, self.example2, self.example3).move_to(self.transformed).shift(0.5 * UP)
        self.result0.move_to(self.result)
        self.set2.move_to(self.set)
        self.transformed2.move_to(self.transformed)

        self.transform_element.next_to(self.arrows[0], RIGHT)
        self.sum.next_to(self.arrows[1], RIGHT)
        self.transform_element2.next_to(self.arrows[2], RIGHT)
    
        self.show_steps()
        self.swap_two_elements()
        self.focus_on_sum()
        self.fade()

    def fade(self):
        self.play(FadeOut(VGroup(
            self.transformed,
            self.arrows[1],
            self.sum,
            self.result,
            self.a,
            self.b,
            self.c,
        )))

    def focus_on_sum(self):
        self.play(
            FadeOut(self.set),
            FadeOut(self.arrows[0]),
            FadeOut(self.arrows[2]),
            FadeOut(self.end),
            FadeOut(self.transform_element),
            FadeOut(self.transform_element2),
        )
        self.remove(self.transformed)
        self.a = self.transformed.copy()
        self.b = self.transformed.copy()
        self.c = self.transformed.copy()
        self.play(
            Transform(self.transformed, self.example0),
            Transform(self.a, self.example1),
            Transform(self.b, self.example2),
            Transform(self.c, self.example3),
        )
        self.play(
            Transform(self.result, self.result0),
        )

    def swap_two_elements(self):
        swap_pairs = [
            (self.set[1], self.set2[3]),
            (self.set[3], self.set2[1]),
            (self.transformed[1], self.transformed2[3]),
            (self.transformed[3], self.transformed2[1]),
        ]
        self.play(*[
            Transform(a, b, path_arc=90 * DEGREES) for a, b in swap_pairs
        ])
        self.play(
            Swap(self.set[1], self.set[5]),
            Swap(self.transformed[1], self.transformed[5]),
        )
    
    def show_steps(self):
        self.play(Write(self.set))
        self.play(
            GrowArrow(self.arrows[0]),
            Write(self.transformed),
            Write(self.transform_element),
        )
        self.play(
            GrowArrow(self.arrows[1]),
            Write(self.sum),
            Write(self.result),
        )
        self.play(
            GrowArrow(self.arrows[2]),
            Write(self.end),
            Write(self.transform_element2),
        )

class FSPool(Scene):
    def construct(self):
        self.title = TextMobject(r"Featurewise Sort Pooling", color=WHITE).scale(1.2)
    
        self.input_set = TexMobject(r"\{3, 1, 2\}").set_color(BLUE).scale(1.2)
        self.sorted_set = TexMobject(r"[1, 2, 3]").set_color(BLUE).scale(1.2)
        self.discrete_weights = TexMobject(r"[0, 1, 0]").set_color(BLUE).scale(1.2)
        self.discrete_weights_updated = TexMobject(r"[0, \frac{2}{3}, \frac{2}{3}, 0]").set_color(GREEN)
        self.result = TexMobject(r"2").set_color(BLUE).scale(1.2)
        self.result_updated = TexMobject(r"3.33\ldots").set_color(GREEN).scale(1.2)
        self.continuous_weights = ContinuousWeights([0, 1, 0]).scale(1.2)

        self.input_set_text = TextMobject("Input").set_color(WHITE).scale(0.8)
        self.sorted_set_text = TextMobject("Sorted").set_color(WHITE).scale(0.8)
        self.discrete_weights_text = TextMobject("Weights").set_color(WHITE).scale(0.8)
        self.result_text = TextMobject("Result").set_color(WHITE).scale(0.8)

        h_spacing = 4
        self.sorted_set.shift(1 * UP)
        self.discrete_weights.move_to(self.sorted_set).shift(3 * DOWN)
        self.discrete_weights_updated.move_to(self.discrete_weights)
        self.input_set.move_to(self.sorted_set).shift(h_spacing * LEFT)
        self.continuous_weights.move_to(self.discrete_weights).shift(h_spacing * LEFT)
        center_sorted_and_weights = (self.sorted_set.get_center() + self.discrete_weights.get_center())/2
        self.result.move_to(center_sorted_and_weights + h_spacing * RIGHT)
        self.result_updated.move_to(self.result).shift(0.5 * RIGHT)

        self.input_set_text.next_to(self.input_set, DOWN)
        self.sorted_set_text.next_to(self.sorted_set, DOWN)
        self.discrete_weights_text.next_to(self.discrete_weights_updated, DOWN)
        self.result_text.next_to(self.result, DOWN)
        
        arrow_args = {
            'color': WHITE,
            'stroke_width': 4,
            'tip_length': DEFAULT_ARROW_TIP_LENGTH / 2,
            'buff': MED_LARGE_BUFF,
        }
        self.arrows = [
            Arrow(self.input_set, self.sorted_set.get_left(), **arrow_args),
            Arrow(self.sorted_set, self.result_updated, **arrow_args),
            Arrow(self.continuous_weights, self.discrete_weights_updated.get_left(), **arrow_args),
            Arrow(self.discrete_weights, self.result_updated, **arrow_args),
        ]
        self.arrow_sort = TextMobject("sort", color=WHITE).scale(0.6).next_to(self.arrows[0], 0.5 * UP)
        self.arrow_dot = TextMobject("dot product", color=WHITE).scale(0.6).next_to(self.result_updated, LEFT).shift(0.3 * LEFT)
        self.arrow_discretise = TextMobject("discretise", color=WHITE).scale(0.6).next_to(self.arrows[2], 0.5 * UP)

        self.show_title()
        self.fixed_size_pass()
        self.different_size_input()
        self.make_piecewise_weights()
        self.show_weight_sampling()
        self.focus_on_weights()
    
    def focus_on_weights(self):
        to_fade = [
            self.input_set,
            self.sorted_set,
            self.discrete_weights_updated,
            self.result_updated,
            self.input_set_text,
            self.sorted_set_text,
            self.discrete_weights_text,
            self.result_text,
            self.arrow_sort,
            self.arrow_dot,
            self.arrow_discretise,
        ] + self.arrows + self.used_objects_in_weight_sampling
        self.play(FadeOut(VGroup(*to_fade)))
        self.remove(self.discrete_weights)
        self.add(self.continuous_weights)

        weights = [
            [0, 0.5, 1, 0.5, 0],
            [0.5, 0.5, 0.5, 0.5, 0.5],
            [0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0],
            [-1, 0, 0, 0, 1],
            [1, -0.5, -0.8, 0.5, -0.2],
        ]
        cws = []
        for w in weights:
            cw = ContinuousWeights(w).scale(1.8).move_to(ORIGIN)
            cw.plane.stretch(0.7, dim=1)
            cw.lines.stretch(0.7, dim=1, about_point=cw.plane.x_axis.get_center())
            cw.add_labels()
            cws.append(cw)
        
        self.add_foreground_mobject(self.continuous_weights.lines)
        self.function_description = TextMobject("Sum").next_to(cws[0], DOWN)

        texts = [
            "Max",
            "Median",
            "Max $-$ Min",
            "???",
        ]

        self.play(
            Transform(self.continuous_weights, cws[0]),
        )
        self.play(
            Write(self.function_description),
            Transform(self.continuous_weights, cws[1]),
        )
        for cw, text in zip(cws[2:], texts):
            text = TextMobject(text).move_to(self.function_description)
            self.play(
                Transform(self.continuous_weights, cw),
                Transform(self.function_description, text),
            )
        self.play(
            FadeOut(self.function_description),
            FadeOut(self.continuous_weights),
            FadeOut(self.title),
        )

    def show_title(self):
        self.title.shift(3 * UP)
        self.play(Write(self.title))
    
    def fixed_size_pass(self):
        self.play(
            Write(self.input_set),
            Write(self.input_set_text),
        )
        self.play(
            GrowArrow(self.arrows[0]),
            Write(self.arrow_sort),
            TransformFromCopy(self.input_set, self.sorted_set),
            Write(self.sorted_set_text),
        )
        self.play(
            Write(self.discrete_weights),
            Write(self.discrete_weights_text),
        )
        result_copy = self.result.copy()
        self.play(
            GrowArrow(self.arrows[1]),
            GrowArrow(self.arrows[3]),
            Write(self.arrow_dot),
            TransformFromCopy(self.sorted_set, self.result),
            TransformFromCopy(self.discrete_weights, result_copy),
            Write(self.result_text),
        )
        self.remove(result_copy)

    def different_size_input(self):
        bigger_set = TexMobject(r"\{3, 1, 4, 2\}").scale(1.2).move_to(self.input_set)
        bigger_sorted = TexMobject(r"[1, 2, 3, 4]").scale(1.2).move_to(self.sorted_set)
        bigger_set.set_color(GREEN)
        bigger_sorted.set_color(GREEN)

        self.play(Transform(self.input_set, bigger_set))
        self.play(Transform(self.sorted_set, bigger_sorted))
    
    def make_piecewise_weights(self):
        self.continuous_weights.plane.stretch(0.7, dim=1)
        self.continuous_weights.lines.stretch(0.7, dim=1, about_point=self.continuous_weights.plane.x_axis.get_center())
        self.continuous_weights.add_labels()
        self.play(
            ShowCreation(self.continuous_weights.plane),
            ShowCreation(self.continuous_weights.nums),
        )
        self.original_discrete_weights = self.discrete_weights.copy()
        self.play(
            Transform(self.discrete_weights, self.continuous_weights.lines)
        )
    
    def show_weight_sampling(self):
        arrows = []
        dots = []
        arrow_template = Arrow(
            ORIGIN,
            DOWN,
            color=BLUE,
            max_stroke_width_to_length_ratio=20,
            max_tip_length_to_length_ratio=0.5,
            stroke_width=20,
        )
        
        new_dots = []
        new_arrows = []
        for coord in self.continuous_weights.evaluate_at(0, 1/3, 2/3, 1):
            dot = Dot(color=GREEN).shift(coord)
            new_dots.append(dot)
            arrow = arrow_template.copy()\
                .set_color(GREEN)\
                .next_to(dot, UP)\
                .set_y(self.continuous_weights.get_top()[1] + MED_SMALL_BUFF + SMALL_BUFF)
            new_arrows.append(arrow)
        self.play(LaggedStart(
            *[GrowArrow(a) for a in new_arrows],
            lag_ratio=0.1,
        ))
        self.play(TransformFromCopy(VGroup(*new_arrows), VGroup(*new_dots)))
        self.play(
            Transform(VGroup(*new_dots), self.discrete_weights_updated),
            Write(self.arrow_discretise),
            GrowArrow(self.arrows[2]),
        )
        self.result_updated.shift(0.3 * LEFT)
        copies = [self.result_updated.copy(), self.result_updated.copy()]
        self.play(
            FadeOut(self.result),
            TransformFromCopy(self.sorted_set, copies[0]),
            TransformFromCopy(self.discrete_weights_updated, copies[1]),
        )
        self.used_objects_in_weight_sampling = new_arrows + new_dots + copies


class Intro(Scene):
    def construct(self):
        self.title_line1 = TextMobject(r"\textbf{FSPool}: Learning Set Representations", stroke_color=WHITE)\
            .scale(1.3)\
            .set_color_by_gradient(BLUE, GREEN)
        self.title_line2 = TextMobject("with Featurewise Sort Pooling", stroke_color=WHITE)\
            .scale(1.3)\
            .next_to(self.title_line1, DOWN)\
            .set_color_by_gradient(BLUE, GREEN)
        self.subtitle = TextMobject(r"\textbf{Yan Zhang}, Jonathon Hare, Adam Prügel-Bennett")\
            .next_to(self.title_line2, DOWN)\
            .set_color(WHITE)
        self.institution = TextMobject("University of Southampton")\
            .next_to(self.subtitle, DOWN)\
            .set_color(WHITE)
        self.title_screen = VGroup(self.title_line1, self.title_line2, self.subtitle, self.institution)
        self.title_line1.shift(UP)
        self.title_line2.shift(UP)
       
        self.pc = load_point_cloud("intro-pointcloud.txt", noise=0.02)\
            .scale(3.5)\
            .sort_points()\
            .thin_out(factor=2)\
            .move_to(ORIGIN)
        self.pc = VGroup(*[
            Dot(color=WHITE, radius=DEFAULT_DOT_RADIUS*0.8).shift(x * RIGHT + y * UP)
            for x, y, z in self.pc.points
        ])
        self.image = ImageMobject("yolo.jpg").scale(1.5).next_to(self.pc, LEFT, buff=LARGE_BUFF)
        pieces = [
            SVGMobject("Chess_tile_nl.svg", fill_color=BLACK),
            SVGMobject("Chess_tile_nl.svg", fill_color=WHITE, stroke_color=DARK_GREY),
        ]
        self.people = VGroup(*pieces)\
            .scale(0.8)\
            .arrange_submobjects()\
            .next_to(self.pc, RIGHT, buff=LARGE_BUFF)

        # animations
        self.title_screen1()
        self.show_examples()
        self.centralise_pc()
        self.complete()

    def complete(self):
        self.play(
            FadeOut(self.pc)
        )

    def show_examples(self):
        self.play(FadeOut(self.title_screen))
        self.play(FadeIn(self.image))
        self.play(FadeIn(self.pc))
        self.play(FadeIn(self.people))

    def title_screen1(self):
        self.add(self.title_screen)

    def centralise_pc(self):
        self.play(
            FadeOut(self.image),
            FadeOut(self.people),
            ApplyMethod(self.pc.move_to, ORIGIN),
        )
        self.remove(self.image, self.people)
        point1 = self.pc.submobjects[4]
        point2 = self.pc.submobjects[50]
        self.play(
            Indicate(point1, scale_factor=2),
            Indicate(point2, scale_factor=2),
            run_time=1.2,
        )
        self.play(Swap(point1, point2))
    

class Responsibility(Scene):
    def construct(self):
        self.ae = Autoencoder()
        self.ae_reference = Autoencoder().set_opacity(0)
        self.add(self.ae)
        self.add(self.ae_reference)
        self.rate = linear

        self.demo_problem()
        self.copy_up_down()
        self.explain_problem()
        self.show_failure()

    def show_failure(self):
        new_input = DottedSquare(color=WHITE, n=12).move_to(self.ae.input_set)
        new_output = DottedSquare(color=WHITE, n=12).move_to(self.ae.output_set)
        palette = [BLUE_E, BLUE_D, BLUE_C, BLUE_B, BLUE_A, TEAL_E, TEAL_D, TEAL_C, TEAL_B, TEAL_A, GREEN_E, GREEN_D, GREEN_C, GREEN_B, GREEN_A, YELLOW_E, YELLOW_D, YELLOW_C, YELLOW_B, YELLOW_A, GOLD_E, GOLD_D, GOLD_C, GOLD_B, GOLD_A, RED_E, RED_D, RED_C, RED_B, RED_A, MAROON_E, MAROON_D, MAROON_C, MAROON_B, MAROON_A, PURPLE_E, PURPLE_D, PURPLE_C, PURPLE_B, PURPLE_A]
        random.shuffle(palette)
        for v, c in zip(new_output.submobjects, palette):
            v.set_color(c)
        self.play(
            Transform(self.ae.input_set, new_input),
            Transform(self.ae.output_set, new_output),
        )
        self.play(ApplyMethod(self.ae.input_set.rotate_to, -PI/4 - 135 * DEGREES), run_time=5, rate_func=linear)
        self.play(
            FadeOut(self.ae),
            FadeOut(self.ae_reference),
        )

    def demo_problem(self):
        # slow rotation phase
        self.play(
            self.animate_rotate_to(-PI/4 - 1.5 * PI, rate_func=self.rate),
            run_time=5,
        )
        self.reset_rotation()


    def copy_up_down(self):
        self.ae_reference.match_style(self.ae)
        self.play(
            ApplyMethod(self.ae_reference.shift, 1.75 * UP),
            ApplyMethod(self.ae.shift, 1.75 * DOWN)
        )
    
    def explain_problem(self):
        # rotate bottom only quickly
        self.play(self.ae.animate_rotate_to(-PI/4))
        self.reset_rotation()
        # slow rotate bottom only
        self.play(
            self.ae.animate_rotate_to(-PI/4 - 3 * PI, rate_func=self.rate),
            run_time=14,
        )
        self.reset_rotation()

    def animate_rotate_to(self, angle, **kwargs):
        return AnimationGroup(
            self.ae.animate_rotate_to(angle, **kwargs),
            self.ae_reference.animate_rotate_to(angle, **kwargs),
        )
    
    def reset_rotation(self):
        self.ae.reset_rotation()
        self.ae_reference.reset_rotation()


class ContinuousWeights(VMobject):
    def __init__(self, weights=None):
        super().__init__()
        self.plane = Axes(
            x_min=0, 
            x_max=2,
            y_min=-1.2,
            y_max=1.2,
            x_axis_config={
                'include_tip': False,
                'include_ticks': False,
                'color': WHITE,
            },
            y_axis_config={
                'include_tip': False,
                'color': WHITE,
            },
        )
        self.weights = weights
        self.plot_weights(weights)
        
        ax = self.plane.create_axis(
            self.plane.y_min, self.plane.y_max, self.plane.y_axis_config
        )
        ax.rotate(90 * DEGREES, about_point=ORIGIN)
        ax.shift(self.plane.x_max * RIGHT)
        self.plane.add(ax)

        self.add(self.plane)
    
    def add_labels(self):
        nums = self.plane.y_axis.get_number_mobjects(
            -1, 0, 1,
            direction=LEFT,
            buff=MED_SMALL_BUFF + SMALL_BUFF,
            number_config={
                'color': WHITE,
            }
        )
        self.nums = nums
        self.add(nums)

    def evaluate_at(self, *xs):
        for x in xs:
            assert 0 <= x <= 1
            float_index = x * (len(self.weights) - 1)
            idx = int(float_index)
            frac = float_index % 1
            if idx == len(self.weights) - 1:
                value = self.weights[-1]
            else:
                left = self.weights[idx]
                right = self.weights[idx + 1]
                value = interpolate(left, right, frac)

            x_coord = self.plane.x_axis.number_to_point(2 * x)[0]
            y_coord = self.plane.y_axis.number_to_point(value)[1]
            yield (x_coord, y_coord, 0)

    def plot_weights(self, weights):
        left, bottom = self.plane.point_to_coords([0, -1, 0])
        right, top = self.plane.point_to_coords([2, 1, 0])

        lines = []
        for i in range(len(weights) - 1):
            start = (i / (len(weights) - 1), weights[i], 0)
            end = ((i + 1) / (len(weights) - 1), weights[i + 1], 0)
            start = (interpolate(left, right, start[0]), interpolate(bottom, top, start[1] / 2 + 0.5), 0)
            end = (interpolate(left, right, end[0]), interpolate(bottom, top, end[1] / 2 + 0.5), 0)
            line = Line(start, end, color=BLUE)
            lines.append(line)
        self.lines = VGroup(*lines)
        self.add(self.lines)


class Autoencoder(VGroup):
    def __init__(self):

        self.input_set = DottedSquare(color=WHITE)

        self.encoder = Trapezoid(2, 1, 1, color=BLUE).rotate(-PI / 2).move_to(1.5 * LEFT)
        self.encoder_text = TextMobject("Encoder", color=WHITE).next_to(self.encoder, DOWN)

        self.arrow0 = Arrow(ORIGIN, RIGHT, color=WHITE).next_to(self.encoder, 0.75*LEFT)

        self.latent = TexMobject(r"\mathbf{z}", color=WHITE)
        self.arrow1 = Arrow(ORIGIN, RIGHT, color=WHITE).next_to(self.latent, 0.75*LEFT)
        self.arrow2 = Arrow(ORIGIN, RIGHT, color=WHITE).next_to(self.latent, 0.75*RIGHT)

        self.decoder = Trapezoid(2, 1, 1, color=BLUE).rotate(PI / 2).move_to(1.5 * RIGHT)
        self.decoder_text = TextMobject("Decoder", color=WHITE).next_to(self.decoder, DOWN)

        self.output_set = DottedSquare(offset_colour=-0.4)
        self.arrow3 = Arrow(ORIGIN, RIGHT, color=WHITE).next_to(self.decoder, 0.75*RIGHT)

        super().__init__(
            self.input_set,
            self.arrow0,
            self.encoder,
            self.encoder_text,
            self.arrow1,
            self.latent,
            self.arrow2,
            self.decoder,
            self.decoder_text,
            self.arrow3,
            self.output_set,
        )

        self.input_set.next_to(self.encoder, 5 * LEFT)
        self.output_set.next_to(self.decoder, 5 * RIGHT)
    

    def animate_rotate_to(self, angle, **kwargs):
        return AnimationGroup(
            ApplyMethod(self.input_set.rotate_to, angle, **kwargs),
            ApplyMethod(self.output_set.rotate_to, angle, **kwargs),
        )
    
    def reset_rotation(self):
        self.input_set.reset_rotation()
        self.output_set.reset_rotation()


class Trapezoid(Polygon):
    def __init__(self, base, top, height, **kwargs):
        points = [
            [-base / 2, 0, 0],
            [ base / 2, 0, 0],
            [ top / 2, height, 0],
            [-top / 2, height, 0],
        ]
        super().__init__(*points, **kwargs)


class DottedSquare(VGroup):
    def __init__(self, tracker=None, angle=PI/4, offset_colour=None, dot_type=Dot, n=4, **kwargs):
        super().__init__(
            *[dot_type(pos, **kwargs) for pos in compass_directions(n, UP)]
        )

        # init rotation correctly
        self.target_angle = ValueTracker(angle)
        self.init_angle = angle
        self.current_angle = 0
        self.rotate(angle)
        # set the offset of where responsibility switches
        if offset_colour is not None:
            self.colourise()
            self.offset_colour = offset_colour
        else:
            self.offset_colour = 0

        def update_rotation(polygon, dt):
            normed_target = keep_in_range(polygon.target_angle.get_value(), PI/2, offset=self.offset_colour)
            normed_current = keep_in_range(self.current_angle, PI/2, offset=self.offset_colour)

            rotation_by = normed_target - normed_current
            polygon.super_rotate(rotation_by)
            self.current_angle = polygon.target_angle.get_value()
            # print(f'{normed_target:.2f} {normed_current:.2f} {rotation_by:.2f}')
            # undo rotation of dots
            for mob in polygon.submobjects:
                mob.rotate(-rotation_by)

        self.add_updater(update_rotation)
    
    def construct_node_tracker(self, index):
        node = self[index]
        color = node.get_color()

        x = DecimalNumber(number=0, include_sign=True)
        y = DecimalNumber(number=0, include_sign=True)

        def updater(d, i, update_color=False):
            old = np.array([x.get_value(), y.get_value(), 0])
            # update value
            new = node.get_center() - self.get_center()
            d.set_value(new[i])
            # check for discontinuity
            if update_color:
                if ((old - new) ** 2).sum() > 0.24:
                    x.set_color(RED)
                    y.set_color(RED)
                else:
                    x.set_color(color)
                    y.set_color(color)

        x.add_updater(lambda d: updater(d, 0, update_color=True))
        y.add_updater(lambda d: updater(d, 1, update_color=False))
        return x, y
    
    def rotate(self, angle):
        self.target_angle.set_value(angle)

    def super_rotate(self, angle):
        super().rotate(angle)

    @property
    def rotate_to(self):
        return self.target_angle.move_to

    def colourise(self):
        colors = [
            RED_D,
            PURPLE_A,
            BLUE_E,
            GOLD_B,
        ]
        for m, c in zip(self, colors):
            m.set_color(c)
        
    def reset_rotation(self):
        self.target_angle.set_value(self.init_angle)
        self.current_angle = self.init_angle
        return self


def keep_in_range(value, max, offset=0):
    # shift the value
    value -= offset
    # scale value into max, then take fractional part
    value = (value / max) % 1
    # rescale back
    value *= max
    # unshift value
    return value + offset


def load_point_cloud(path, noise=0):
    points = []
    with open(path) as fd:
        for line in fd:
            _, y, x = line.split(" ")
            x, y = map(float, [x, y])
            points.append([x, 1 - y, 0])
    points = np.array(points)
    if noise > 0:
        points += noise * np.random.randn(*points.shape)
    mob = PMobject()
    mob.add_points(points)
    return mob
