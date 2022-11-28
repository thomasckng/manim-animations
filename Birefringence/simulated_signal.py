from manim import *
import numpy as np
import matplotlib.pyplot as plt
import lalsimulation as lalsim
import lal
from scipy import interpolate

class amplitude_frequency_plot(Scene):
    def construct(self):
        # frequency array parameters
        df = 0.25
        f_min = 10
        f_max = 2030+df
        f_ref = 100

        approximant = lalsim.SimInspiralGetApproximantFromString("IMRPhenomD")

        # source parameters
        m1_msun = 10
        m2_msun = 10
        chi1 = [0, 0, 0]
        chi2 = [0, 0, 0]
        dist_mpc = 1000
        inclination = np.pi * 0.9
        phi_ref = 0

        m1_kg = m1_msun*lal.MSUN_SI
        m2_kg = m2_msun*lal.MSUN_SI
        distance = dist_mpc*1e6*lal.PC_SI

        hp, hc = lalsim.SimInspiralChooseFDWaveform(m1_kg, m2_kg,
                                                    chi1[0], chi1[1], chi1[2],
                                                    chi2[0], chi2[1], chi2[2],
                                                    distance, inclination,
                                                    phi_ref, 0, 0., 0.,
                                                    df, f_min, f_max, f_ref,
                                                    None, approximant)

        hl = (hp.data.data + (hc.data.data * 1j)) /np.sqrt(2)
        hr = (hp.data.data - (hc.data.data * 1j)) /np.sqrt(2)

        freq = np.arange(f_min, f_max, df)
        # freq = np.log10(freq)

        hl = hl[int(f_min/df):int(f_max/df)]
        hr = hr[int(f_min/df):int(f_max/df)]

        hl = interpolate.interp1d(freq, hl)
        hr = interpolate.interp1d(freq, hr)
        
        kappa = 0.5

        self.axes2 = Axes(x_range=[0,5], y_range=[0,2],axis_config={"include_tip": True, "include_numbers": False, "include_ticks":True}).scale(0.9)

        self.axes = Axes(x_range=[1, 3.5], y_range=[-38, -20, 3],
                            axis_config={"include_tip": True, "include_numbers": True, "scaling":LogBase(10)}).scale(0.9)
        y_label = self.axes.get_y_axis_label(Tex(r"$|\tilde{h}|$"))
        x_label = self.axes.get_x_axis_label(Tex(r"$f$ (Hz)"))
        self.grid_labels = VGroup(x_label, y_label)
        self.add(self.axes2, self.grid_labels)
        self.play(Transform(self.axes2,self.axes))

        self.original_left = self.axes.plot(lambda x : abs(hl(x)), x_range=[np.log10(f_min),np.log10(f_max-df*2)], color=BLUE)
        self.original_right = self.axes.plot(lambda x : abs(hr(x)), x_range=[np.log10(f_min),np.log10(f_max-df*2)], color=RED)


        original_birefringence_left_0 = lambda x: abs(hl(x) * np.exp(-kappa*(dist_mpc/1000)*(x/100)))
        original_birefringence_right_0 = lambda x: abs(hr(x) * np.exp(kappa*(dist_mpc/1000)*(x/100)))
        self.original_birefringence_left_0 = self.axes.plot(lambda x : original_birefringence_left_0(x), x_range=[np.log10(f_min), np.log10(f_max-df*2)], color=BLUE)
        self.original_birefringence_right_0 = self.axes.plot(lambda x : original_birefringence_right_0(x), x_range=[np.log10(f_min), np.log10(f_max-df*2)], color=RED)
        kappa_0 = Tex("Birefringence")
        kappa_0.shift(np.array([-1.5, -1.5, 0]))

        # original_birefringence_left_1 = lambda x: abs(hl(x))
        # original_birefringence_right_1 = lambda x: abs(hr(x))
        # self.original_birefringence_left_1 = self.axes.plot(lambda x : original_birefringence_left_1(x), x_range=[np.log10(f_min), np.log10(f_max-df*2)], color=BLUE)
        # self.original_birefringence_right_1 = self.axes.plot(lambda x : original_birefringence_right_1(x), x_range=[np.log10(f_min), np.log10(f_max-df*2)], color=RED)
        kappa_1 = Tex("GR")
        kappa_1.shift(np.array([-1.5, -1.5, 0]))

        # original_birefringence_left_2 = lambda x: abs(hl(x) * np.exp(kappa*(dist_mpc/1000)*(x/100)))
        # original_birefringence_right_2 = lambda x: abs(hr(x) * np.exp(-kappa*(dist_mpc/1000)*(x/100)))
        # self.original_birefringence_left_2 = self.axes.plot(lambda x : original_birefringence_left_2(x), x_range=[np.log10(f_min), np.log10(f_max-df*2)], color=BLUE)
        # self.original_birefringence_right_2 = self.axes.plot(lambda x : original_birefringence_right_2(x), x_range=[np.log10(f_min), np.log10(f_max-df*2)], color=RED)
        # kappa_2 = Tex("Birefringence (right-handed)")
        # kappa_2.shift(np.array([-1.5, -1.5, 0]))

        self.play(Create(self.original_left), Create(self.original_right), Write(kappa_1))
        self.wait()
        self.play(ReplacementTransform(kappa_1, kappa_0))
        self.play(ReplacementTransform(self.original_left, self.original_birefringence_left_0),
                    ReplacementTransform(self.original_right, self.original_birefringence_right_0),
                    run_time = 5)
        self.wait(3)
        # self.play(ReplacementTransform(self.original_birefringence_left_0, self.original_birefringence_left_1),
        #             ReplacementTransform(self.original_birefringence_right_0, self.original_birefringence_right_1),
        #             ReplacementTransform(kappa_0, kappa_1),
        #             run_time = 2)
        # self.wait()
        # self.play(ReplacementTransform(self.original_birefringence_left_1, self.original_birefringence_left_2),
        #             ReplacementTransform(self.original_birefringence_right_1, self.original_birefringence_right_2),
        #             ReplacementTransform(kappa_1, kappa_2),
        #             run_time = 2)
        # self.wait()
        self.play(Uncreate(self.original_birefringence_left_0), Uncreate(self.original_birefringence_right_0),
                    Uncreate(kappa_0))
        self.wait()