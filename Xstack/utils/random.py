#!/usr/bin/env python3
"""
====================================
Module for bootstrap & randomization
====================================
:Authors:   Shi-Jiang Chen (MPE, USTC)
            Johannes Buchner (MPE)
            Teng Liu (USTC)
:Email:     JohnnyCsj666@gmail.com


"""
import numpy as np


def calc_bootstrap_weights(Nsrc,bootstrap_portion=1.0,rng=None):
    """
    Generate bootstrap weights (see examples for clarification).

    Parameters
    ----------
    Nsrc : int
        Number of sources in the list.
    bootstrap_portion : float, optional
        The portion of sources to be drawn in each bootstrap realization. 
        Defaults to 1.0, which means drawing the same number of sources as 
        the original list (with replacement).
    rng : numpy.random._generator.Generator, optional
        The random number generator to use. If None, a new default generator 
        will be created. Defaults to None.
    
    Returns
    -------
    w : numpy.ndarray
        Bootstrap weights list.
        
    Examples
    --------
    - Let's say we have 20 sources, and we want to do 1 bootstrap realizations.

      .. code-block:: python
  
          btw_lst = calc_bootstrap_weights(20)
  
      ``btw_lst`` is just how many times each source appear in the current realization:
  
      .. code-block:: python
  
          array([0, 0, 0, 2, 2, 0, 1, 1, 2, 1, 3, 0, 1, 2, 1, 0, 0, 0, 2, 2])
  
      And we know that:

      - 0-th source appears 0 time.
      - 1-st source appears 0 time.
      - 2-nd source appears 0 time.
      - 3-rd source appears 2 times.
      - ...

    - If we want to do 10 bootstrap realizations:

      .. code-block:: python
  
          btw_lst_rlz = [
              calc_bootstrap_weights(20)
              for _ in range(10)
          ]   # realizations of bootstrap weight list, (num_bootstrap, Nsrc)
  
      This gives us:
  
      .. code-block:: python
  
          [array([0, 0, 1, 2, 0, 2, 1, 1, 0, 2, 1, 2, 0, 1, 2, 1, 1, 1, 0, 2]),
          array([0, 1, 1, 1, 2, 4, 1, 0, 0, 2, 2, 3, 2, 0, 0, 0, 0, 1, 0, 0]),
          array([1, 1, 1, 1, 3, 2, 1, 2, 0, 1, 0, 1, 0, 1, 0, 0, 3, 1, 0, 1]),
          array([0, 5, 1, 0, 0, 3, 0, 1, 1, 0, 1, 0, 3, 0, 1, 1, 0, 3, 0, 0]),
          array([1, 1, 2, 0, 1, 1, 2, 0, 0, 1, 0, 2, 1, 2, 1, 2, 0, 3, 0, 0]),
          array([2, 0, 2, 2, 0, 1, 0, 3, 1, 0, 1, 0, 1, 1, 0, 1, 2, 1, 2, 0]),
          array([2, 0, 1, 0, 1, 1, 1, 2, 1, 0, 2, 1, 1, 0, 3, 1, 0, 2, 1, 0]),
          array([0, 1, 2, 0, 1, 0, 0, 1, 2, 0, 1, 1, 1, 2, 1, 1, 1, 1, 2, 2]),
          array([2, 2, 1, 0, 0, 1, 1, 2, 2, 1, 1, 1, 0, 1, 2, 0, 1, 1, 1, 0]),
          array([3, 4, 0, 1, 3, 2, 0, 0, 2, 0, 1, 0, 0, 2, 0, 0, 1, 0, 0, 1])]

    """
    if rng is None:
        rng = np.random.default_rng()

    M = int(Nsrc*bootstrap_portion)
    p = np.full(Nsrc,1.0/Nsrc)
    w = rng.multinomial(M,p)

    return w