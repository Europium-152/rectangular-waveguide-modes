"""Header texts of the Alimenti et al. (1994) files (card L2)."""
SRC = ('F. Alimenti, M. Mongiardo, R. Sorrentino, "Design of mitered H-plane bends in rectangular waveguides by '
       'combined mode matching and finite differences", Proc. 24th European Microwave Conference (1994), '
       'vol. 1, pp. 289-293, doi:10.1109/EUMA.1994.337223')
GEOM_T = ('t = length of the outer wall that is left between the end of the 45 deg miter and the plane of the other '
          "arm's inner wall (dimension arrows of the inset of Fig. 4, text p. 291; the arrows end on the miter "
          'corner and on the thin line that continues the inner wall). The cut along each outer wall, measured from '
          'the removed outer corner, is c = a - t (Ma et al. 1997 c), and the CONVENTIONS.md section 6 chamfer '
          'distance from the inner corner is d/a = (1 + t/a)/sqrt(2): t/a = 0 is the fully mitered bend of Fig. 3 '
          '(d/a = 1/sqrt(2), the optical mirror), t/a = 1 the unmitered corner (d/a = sqrt(2)).')
DFA = lambda t: (1 + t) / 2 ** 0.5
FIG = {
    2: dict(figure='Fig. 2, p. 290 (PDF p. 2): "Reflection coefficient of a H-plane corner in rectangular waveguide" '
                   '(unmitered right-angle H-plane corner), MM (mode matching, line with dots) and MMFD (line); MMFD '
                   'with M = 5 TEn0 modes and a 200x200 finite-difference mesh (text p. 290)',
            axes='x = normalized frequency f/fc (fc = TE10 cut-off), linear, axis 1..2; y = |S11| in dB, linear, '
                 'axis 0..-10 dB',
            geom='unmitered 90 deg H-plane corner (inset: square corner, both arms of width a): t/a = 1 in the '
                 'convention of Figs. 4-6, d/a = sqrt(2) = 1.414 (CONVENTIONS.md section 6)'),
    3: dict(figure='Fig. 3, p. 291 (PDF p. 3): "Reflection coefficient of a fully mitered H-plane 90 deg bend", MM '
                   '(line) and MMFD (line with diamonds)',
            axes='x = f/fc, linear, axis 1..2; y = |S11| in dB, linear, axis 0..-70 dB',
            geom='fully mitered 90 deg H-plane bend (inset: the miter joins the two outer-wall points that face the '
                 'inner corner): t/a = 0, d/a = 1/sqrt(2) = 0.707, the optical mirror (CONVENTIONS.md section 6 plain '
                 'miter)'),
    4: dict(figure='Fig. 4, p. 291 (PDF p. 3): "Reflection coefficient of mitered H-plane right angle bends in '
                   'rectangular waveguide for various values of the ratio t/a" (MMFD)',
            axes='x = f/fc, linear, axis 1..2; y = |S11| in dB, linear, axis 0..-50 dB; the reflection zeros lie '
                 'below the frame',
            geom='90 deg H-plane bend, 45 deg miter; t/a = 0.30, 0.34, 0.38, i.e. d/a = 0.919, 0.948, 0.976'),
    5: dict(figure='Fig. 5, p. 292 (PDF p. 4): "Design chart giving the normalized matching frequency and the '
                   'normalized 30dB bandwidth versus the mitering ratio t/a" (normalization by the cut-off frequency)',
            axes='x = t/a, linear, axis 0.30..0.40; matching frequency f/fc on the left axis 1.0..2.0; 30 dB '
                 'bandwidth / fc on the right axis 0..0.5 (same gridlines: bandwidth = (left value - 1)/2); the '
                 'curves are identified by the arrows printed next to them',
            geom='90 deg H-plane bend, 45 deg miter; d/a = (1 + t/a)/sqrt(2) = 0.919..0.990'),
    6: dict(figure='Fig. 6, p. 292 (PDF p. 4): "Reflection coefficient of a H-plane 90 deg mitered bend for '
                   't/a=0.386", MMFD (line) and BCMM (Reiter & Arndt 1994, line with diamonds); dotted line at '
                   '-30 dB',
            axes='x = f/fc, linear, axis 1..2; y = |S11| in dB, linear, axis 0..-40 dB; the zeros lie below the '
                 'frame',
            geom='90 deg H-plane bend, 45 deg miter, t/a = 0.386, d/a = 0.980'),
}
FILES = {
    2: {'alimenti1994_fig2_mmfd.csv': (['MMFD'], 'MMFD = the line; the MM line is printed on top of it and is not '
                                       'distinguishable from it anywhere (agreement within the stroke width, about '
                                       '0.1 dB), so this curve stands for both'),
        'alimenti1994_fig2_mm.csv': (['MM markers'], 'MM markers = the 10 dots of the MM line (one row per dot)')},
    3: {'alimenti1994_fig3_mm.csv': (['MM'], 'MM = the line; the MMFD line coincides with it within the stroke '
                                     'width up to f/fc = 1.93 (-24 dB) and is digitized with it; beyond, the two '
                                     'lines separate in a reflection zero at f/fc = 1.98 that reaches the -70 dB '
                                     'frame (not digitized: two crossing near-vertical strokes that cannot be '
                                     'assigned)'),
        'alimenti1994_fig3_mmfd.csv': (['MMFD markers'], 'MMFD markers = the 10 diamonds of the MMFD line (one row '
                                       'per symbol); MMFD line: see alimenti1994_fig3_mm.csv')},
    4: {'alimenti1994_fig4_t030.csv': (['t/a=0.30'], 't/a=0.30 = thin solid line; zero below the frame near '
                                       'f/fc = 1.29'),
        'alimenti1994_fig4_t034.csv': (['t/a=0.34'], 't/a=0.34 = solid line with dots (dots not digitized '
                                       'separately: line-style symbols on the line); zero below the frame near f/fc = 1.50'),
        'alimenti1994_fig4_t038.csv': (['t/a=0.38'], 't/a=0.38 = dashed line with open squares (squares left '
                                       'out: line-style symbols); zeros near f/fc = 1.69 (below the frame) and '
                                       '1.967 (below about -42 dB, where the dashes of both flanks end)')},
    5: {'alimenti1994_fig5_matching_frequency.csv': (['matching frequency'], 'matching frequency = the line '
                                                    'marked with the left arrow (frequency of the reflection zero, '
                                                    'f/fc, left axis)'),
        'alimenti1994_fig5_bandwidth.csv': (['bandwidth, one zero', 'bandwidth, two zeros'],
                                            '30 dB bandwidth / fc (right axis, the line marked with the right arrow); '
                                            'it jumps at t/a = 0.3867 (vertical stroke): "bandwidth, one zero" ends '
                                            'and "bandwidth, two zeros" starts at the jump abscissa with the lower '
                                            'and upper ends of the vertical stroke')},
    6: {'alimenti1994_fig6_mmfd.csv': (['MMFD'], 'MMFD = thin line; three branches separated by zeros below the '
                                       'frame (between f/fc 1.68 and 1.775, and at 1.943 +- 0.006); for f/fc < 1.65 MMFD and BCMM coincide '
                                       'within the stroke width and the same points are given in both files; peak '
                                       'of the middle lobe touches the dotted -30 dB line and was read from the '
                                       'lower edge of the stroke'),
        'alimenti1994_fig6_bcmm.csv': (['BCMM', 'BCMM markers'], 'BCMM = thin line with diamonds (Reiter & Arndt '
                                       '1994, their ref. [9]); BCMM markers = the 10 diamonds (one row per '
                                       'symbol); for f/fc < 1.65 coincident with MMFD (same points)')},
}
