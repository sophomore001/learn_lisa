#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# BSD 3-Clause License
#
# Copyright 2022, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# This software may be subject to U.S. export control laws. By accepting this
# software, the user agrees to comply with all applicable U.S. export laws and
# regulations. User has the responsibility to obtain export licenses, or other
# export authority as may be required before exporting such information to
# foreign countries or providing access to foreign persons.
#
"""
Generate figures for various glitch types.

This script is used by Gitlab-CI to generate example figures.

Authors:
    Jean-Baptiste Bayle <j2b.bayle@gmail.com>
"""

import logging
import argparse
import lisaglitch


def main():
    """Main function."""
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger('lisaglitch').setLevel(logging.DEBUG)

    generators = {
        'step': step,
        'rectangle': rectangle,
        'one-sided-double-exp': one_sided_double_exp,
        'integrated-one-sided-double-exp': integrated_one_sided_double_exp,
        'two-sided-double-exp': two_sided_double_exp,
        'integrated-two-sided-double-exp': integrated_two_sided_double_exp,
        'shapelet': shapelet,
        'integrated-shapelet': integrated_shapelet,
        'lpf-library': lpf_library,
        'lpf-model-library': lpf_model_library,
        'hdf5': hdf5,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('glitch', help='glitch type', choices=generators.keys())
    args = parser.parse_args()

    glitch = generators[args.glitch]()
    glitch.plot(f'{args.glitch}.pdf')
    glitch.write(f'{args.glitch}.h5')


def step():
    """Generate a step glitch."""
    return lisaglitch.StepGlitch(inj_point='tm_12')


def rectangle():
    """Generate a rectangle glitch."""
    return lisaglitch.RectangleGlitch(inj_point='tm_12', width=2, t0=2032084800.0)
    #return lisaglitch.RectangleGlitch(inj_point='tm_12', width=2, size= 40000, t0=2032084800.0, t_inj = 2032085800.0)


def one_sided_double_exp():
    """Generate a one-sided double-exponential glitch."""
    return lisaglitch.OneSidedDoubleExpGlitch(
        inj_point='tm_12', t_rise=1, t_fall=2)


def integrated_one_sided_double_exp():
    """Generate an integrated one-sided double-exponential glitch."""
    return lisaglitch.IntegratedOneSidedDoubleExpGlitch(
        inj_point='tm_12', t_rise=1, t_fall=2)


def two_sided_double_exp():
    """Generate a two-sided double-exponential glitch."""
    return lisaglitch.TwoSidedDoubleExpGlitch(
        inj_point='tm_12', t_rise=1, t_fall=2, displacement=10)


def integrated_two_sided_double_exp():
    """Generate an integrated two-sided double-exponential glitch."""
    return lisaglitch.IntegratedTwoSidedDoubleExpGlitch(
        inj_point='tm_12', t_rise=1, t_fall=2, displacement=10)


def shapelet():
    """Generate a shapelet glitch."""
    return lisaglitch.ShapeletGlitch(inj_point='tm_12')


def integrated_shapelet():
    """Generate an integrated shapelet glitch."""
    return lisaglitch.IntegratedShapeletGlitch(inj_point='tm_12')


def lpf_library():
    """Generate a glitch from the default LPF library."""
    return lisaglitch.LPFLibraryGlitch(
        path='data/lpf-glitch-library.h5',
        run=66,
        glitch=3,
        inj_point='tm_12')


def lpf_model_library():
    """Generate a glitch from the default LPF model library."""
    return lisaglitch.LPFLibraryModelGlitch(
        path='data/lpf-glitch-library.h5',
        run=66,
        glitch=3,
        inj_point='tm_12')


def hdf5():
    """Generate a glitch from an HDF5 file."""
    return lisaglitch.HDF5Glitch(
        path='data/lpf-glitch-library.h5',
        node='timeseries/run09/glitch06',
        inj_point='tm_12')


if __name__ == '__main__':
    main()
