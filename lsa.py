"""
CMB Large Scale Anomaly estimators, including:
- Lack of correlation
- Quadrupole-Octupole alignment
- Odd-parity asymmetry
- Hemispherical power asymmetry

[Shi et al. 2023, doi: 10.3847/1538-4357/acb339]
"""

import numpy as np
import numpy.typing as npt
from scipy.special import legendre, factorial
import scipy.optimize as sop
import healpy as hp

#############
### Tools ###
#############


def convert_Emap_naive(m: npt.NDArray) -> npt.NDArray:
    '''
    Convert the map to E-mode map using the naive method.
    Equation (2) in the Shi et al. 2023.

    Parameters
    ----------
    m : npt.NDArray

    Returns
    -------
    Emap : npt.NDArray
    '''
    nside = hp.get_nside(m[0])
    _, almE, _ = hp.map2alm(m)
    Emap = hp.alm2map(almE, nside=nside, verbose=False)
    return Emap


def convert_Bmap_naive(m: npt.NDArray) -> npt.NDArray:
    '''
    Convert the map to B-mode map using the naive method.

    Parameters
    ----------
    m : npt.NDArray

    Returns
    -------
    Bmap : npt.NDArray
    '''

    nside = hp.get_nside(m[0])
    _, _, almB = hp.map2alm(m)
    Bmap = hp.alm2map(almB, nside=nside, verbose=False)
    return Bmap


###########################
### Lack of correlation ###
###########################

def calc_SXY_2pt(cosangs: npt.NDArray,
                 corr: npt.NDArray,
                 ulim: float | npt.NDArray) -> npt.NDArray:
    '''
    Estimator for the lack of correlation.
    Compute the estimator by doing numerical integration of 2-pt correlation function.
    Equation (10) in Shi et al. 2023.

    Parameters
    ----------
    cosangs : array
        Cosine of angle separation.
    corr : array
        2-pt correlation function.
    ulim : float or 1D array
        Upper limit(s) of the integral (cosine of angle separation).
        The lower limit is always -1.

    Return
    ------
    SXY : array
    '''
    ulim = np.atleast_1d(ulim)
    cor_tot = []
    for costheta in ulim:
        dcostheta = np.diff(cosangs[cosangs < costheta])
        cor = np.delete(corr[:, cosangs < costheta], 0, axis=1)
        cor_tot.append(np.sum(cor**2 * dcostheta, axis=1))
    return np.array(cor_tot)


def calc_S12_aps(cl: npt.NDArray,
                 lmin: int = 2,
                 lmax: int = 10,
                 mode: str = 'TT',
                 llim: float = 0.5) -> float:
    '''
    Estimator for the lack of correlation, S_{1/2}.
    Compute the lack of correlation estimator using the angular power spectra.
    Equation (11) in Shi et al. 2023.

    Parameters
    ----------
    cl : array
        The angular power spectrum.
    lmin, lmax : int
        The minimum and maximum index of summation.
    mode : str
        'TT' or 'EE'.
    llim : float
        The lower limit of the integral (multipole).

    Return
    ------
    S12 : float
    '''
    supported_modes = ['TT', 'EE']
    if mode not in supported_modes:
        raise Exception(
            f'mode: {mode} not yet considered. Supported modes: {supported_modes}')

    ell = np.arange(lmin, lmax + 1)
    if mode == 'TT':
        ps = ((2 * ell + 1) / (4 * np.pi)) * cl[lmin:lmax + 1]
    else:
        ps = ((2 * ell + 1) / (4 * np.pi)) * \
            (factorial(ell + 2) / factorial(ell - 2)) * cl[lmin:lmax + 1]

    return ps.dot(np.array(np.matmul(calc_Imat(llim, lmin, lmax), ps)).ravel())


def calc_Imat(x: float,
              lmin: int = 2,
              lmax: int = 10) -> npt.NDArray:
    '''
    Compute the coupling matrix.
    Equation (12) in Shi et al. 2023.
    Derivation can be found in the appendix of doi:10.1111/j.1365-2966.2009.15270.x.

    Parameters
    ----------
    x : float
        The lower limit of the integral.
    lmin, lmax : int
        minimum / maximum multipole to include.

    Return
    ------
    The coupling matrix.
    '''
    Imat = np.zeros((lmax + 1, lmax + 1))
    Imat[0][0] = x + 1
    Imat[1][1] = (x**3. + 1) / 3

    # generate legendre array
    Pmat = []
    for i in range(lmax + 2):
        Pmat.append(legendre(i)(x))

    # the first row
    for n in range(1, lmax + 1):
        Imat[0][n] = (-n * Pmat[0] * (Pmat[n - 1] -
                      x * Pmat[n])) / (n * (n + 1))
    # off-diagonal term
    for m in range(1, lmax + 1):
        for n in range(m + 1, lmax + 1):
            Imat[m][n] = (m * Pmat[n] * (Pmat[m - 1] - x * Pmat[m]) - n * Pmat[m]
                          * (Pmat[n - 1] - x * Pmat[n])) / (n * (n + 1) - m * (m + 1))

    Imat = Imat + Imat.transpose()
    # diagonal term
    Imat[0][0] = x + 1
    Imat[1][1] = (x**3. + 1) / 3
    for n in range(2, lmax):
        Imat[n][n] = ((Pmat[n + 1] - Pmat[n - 1]) * (Pmat[n] - Pmat[n - 2]) - (2 * n - 1) * Imat[n + 1]
                      [n - 1] + (2 * n + 1) * Imat[n][n - 2] + (2 * n - 1) * Imat[n - 1][n - 1]) / (2 * n + 1)
    n = lmax  # last term
    Inew = ((n + 1) * Pmat[n - 1] * (Pmat[n] - x * Pmat[n + 1]) - (n - 1)
            * Pmat[n + 1] * (Pmat[n - 2] - x * Pmat[n - 1])) / (-4 * n - 2)
    Imat[n][n] = ((Pmat[n + 1] - Pmat[n - 1]) * (Pmat[n] - Pmat[n - 2]) - (2 * n - 1) * \
                  Inew + (2 * n + 1) * Imat[n][n - 2] + (2 * n - 1) * Imat[n - 1][n - 1]) / (2 * n + 1)

    return np.array(Imat)[lmin:, lmin:]


#######################################
### Quadrupole - Octupole alignment ###
#######################################

def calc_QO(alm_in1: npt.NDArray,
            alm_in2: npt.NDArray | None = None,
            nside: int = 64,
            pol: bool = True) -> npt.NDArray:
    '''
    Estimator for the quadrupole - octupole alignment.
    In the current version, if do polarization, the direction of temperature is used as
    the polarization direction.
    Equation (16) in Shi et al. 2023.

    Parameters
    ----------
    alm_in1, alm_in2 : array
        Should match the format as the return of hp.map2alm().
        Will be converted to 2d array if it's not.
    nside : int
        HEALPix NSIDE.
    pol : bool
        Polarization option.

    Return
    ------
    if pol==True:
        L23TT, L23EE, L23BB, unit vector.
    else:
        L23TT, unit vector.
    '''
    if alm_in2 is None:
        alm_in2 = alm_in1.copy()
    alm_in1, alm_in2 = np.atleast_2d(alm_in1, alm_in2)

    L23TT, ipix = find_max_L23_ipix(alm_in1[0], alm_in2[0], nside=nside)
    if not pol:
        return L23TT, np.array(hp.pix2vec(nside, ipix))
    else:
        theta, phi = hp.pix2ang(nside, ipix)
        L23EE = calc_L23(
            alm_in1[1],
            alm_in2[1],
            psi=-phi,
            theta=-theta,
            phi=0.)
        L23BB = calc_L23(
            alm_in1[2],
            alm_in2[2],
            psi=-phi,
            theta=-theta,
            phi=0.)
        return np.array([L23TT, L23EE, L23BB]), np.array(
            hp.pix2vec(nside, ipix))


def angmmt_nume(alm1: npt.NDArray, alm2: npt.NDArray) -> float:
    '''
    Compute the numerator in the angular momentum expression.
    See definition in Equation (15) in Shi et al. 2023.

    Parameters
    ----------
    alm_in1, alm_in2 : array
        The alm at fixed ell.
        Should have length ell+1.

    Return
    ------
    The numerator in Eq. (15).
    '''
    assert len(alm1) == len(alm2)
    m = np.arange(1, len(alm1))
    return 2 * np.sum(m**2 * np.abs(np.conj(alm1[1:]) * alm2[1:]))


def angmmt_deno(alm1: npt.NDArray, alm2: npt.NDArray) -> float:
    '''
    Compute the denominator in the angular momentum expression.
    See definition in Equation (15) in Shi et al. 2023.

    Parameters
    ----------
    alm_in1, alm_in2 : array
        The alm at fixed ell.
        Should have length ell+1.

    Return
    ------
    The denominator in Eq. (15).
    '''
    assert len(alm1) == len(alm2)
    l = len(alm1) - 1
    return l**2 * (2 * np.sum(np.abs(np.conj(alm1) * alm2)) -
                   np.abs(np.conj(alm1[0]) * alm2[0]))


def calc_L23(alm_in1: npt.NDArray,
             alm_in2: npt.NDArray,
             psi: float = 0.,
             theta: float = 0.,
             phi: float = 0.) -> float:
    '''
    Calculate the L23 estimator value.
    Equation (16) in Shi et al. 2023.

    Parameters
    ----------
    alm_in1, alm_in2 : 1D array
        Should match the format as the return of hp.map2alm().
    psi, theta : float
        The angles to rotate the alm.

    Return
    ------
    L23TT in Eq. (16).
    '''
    ll, _ = hp.Alm.getlm(lmax=hp.Alm.getlmax(len(alm_in1)))

    alm1 = alm_in1.copy()
    hp.rotate_alm(alm1, psi=psi, theta=theta, phi=phi)

    alm2 = alm_in2.copy()
    hp.rotate_alm(alm2, psi=psi, theta=theta, phi=phi)

    nume2 = angmmt_nume(alm1[ll == 2], alm2[ll == 2])
    deno2 = angmmt_deno(alm1[ll == 2], alm2[ll == 2])

    nume3 = angmmt_nume(alm1[ll == 3], alm2[ll == 3])
    deno3 = angmmt_deno(alm1[ll == 3], alm2[ll == 3])

    L23 = 0.5 * (nume2 / deno2 + nume3 / deno3)
    return L23


def find_max_L23_ipix(alm_in1: npt.NDArray,
                      alm_in2: npt.NDArray | None = None,
                      nside: int = 64) -> int:
    '''
    Find the maximum L23 estimator value and the corresponding pixel index.

    Parameters
    ----------
    alm_in1, alm_in2 : array
        Should match the format as the return of hp.map2alm().
        If `alm_in2` is None, it will be set to `alm_in1`.
    nside : int
        HEALPix NSIDE, the resolution of the map to use/

    Return
    ------
    L23 : float
        The maximum L23 value.
    ipix : int
        The pixel index of the maximum L23 value.
    '''
    N = int(np.log2(nside))
    if alm_in2 is None:
        alm_in2 = alm_in1.copy()

    L23 = []
    for pix in range(hp.nside2npix(1)):
        theta, phi = hp.pix2ang(1, pix)
        L23.append(calc_L23(alm_in1, alm_in2, psi=-phi, theta=-theta, phi=0.))
    if nside == 1:
        return np.max(L23), np.argmax(L23)

    nside_lo = 1
    ipix_hi = np.arange(12)
    L23_lo = np.array(L23)
    for nside_hi in [2**i for i in range(1, N + 1)]:
        ipix_lo = ipix_hi[np.argsort(L23_lo)[-4:]]  # get the top 4 pixels
        ipix_lo_nest = hp.ring2nest(nside_lo, ipix_lo)
        ipix_hi_nest = (4 *
                        ipix_lo_nest[np.newaxis, :] +
                        np.arange(4)[:, np.newaxis]).ravel()
        ipix_hi = hp.nest2ring(nside_hi, ipix_hi_nest)
        L23_hi = []
        for pix in ipix_hi:
            theta, phi = hp.pix2ang(nside_hi, pix)
            L23_hi.append(
                calc_L23(
                    alm_in1,
                    alm_in2,
                    psi=-phi,
                    theta=-theta,
                    phi=0.))
        nside_lo = nside_hi
        L23_lo = np.array(L23_hi)
    return np.max(L23_lo), ipix_hi[np.argmax(L23_hi)]


############################
### Odd-parity asymmetry ###
############################

def calc_R(cl: npt.NDArray,
           lmin: int = 2,
           lmax: int | None = None) -> npt.NDArray:
    '''
    The ratio estimator for the odd-parity asymmetry.
    Equation (17) in Shi et al. 2023.

    Parameters
    ----------
    cl : array
        The angular power spectrum, default to start with ell=0.
    lmin, lmax : int
        The minimum/maximum of the summation.
        Including lmax.

    Return
    ------
    The ratio estimator.
    '''
    _cl = np.atleast_2d(cl)
    if not lmax:
        lmax = len(_cl[0]) + 1

    if lmin % 2:  # lmin odd, '-' case
        C_m = calc_C(_cl[:, lmin:lmax + 1:2], lmin, lmax)
        C_p = calc_C(_cl[:, lmin + 1:lmax + 1:2], lmin + 1, lmax)
        return C_p / C_m
    else:  # lmin even, '+' case
        C_m = calc_C(_cl[:, lmin + 1:lmax + 1:2], lmin + 1, lmax)
        C_p = calc_C(_cl[:, lmin:lmax + 1:2], lmin, lmax)
        return C_p / C_m


def calc_D(cl: npt.NDArray,
           lmin: int = 2,
           lmax: int | None = None) -> npt.NDArray:
    '''
    The difference estimator for the odd-parity asymmetry.
    Equation (18) in Shi et al. 2023.

    Parameters
    ----------
    cl : array
        The angular power spectrum, default to start with ell=0.
    lmin, lmax : int
        The minimum/maximum of the summation.
        Including lmax.

    Return
    ------
    The difference estimator.
    '''
    _cl = np.atleast_2d(cl)
    if not lmax:
        lmax = len(_cl[0]) + 1

    if lmin % 2:  # lmin odd, '-' case
        C_m = calc_C(_cl[:, lmin:lmax + 1:2], lmin, lmax)
        C_p = calc_C(_cl[:, lmin + 1:lmax + 1:2], lmin + 1, lmax)
        return C_p - C_m
    else:  # lmin even
        C_m = calc_C(_cl[:, lmin + 1:lmax + 1:2], lmin + 1, lmax)
        C_p = calc_C(_cl[:, lmin:lmax + 1:2], lmin, lmax)
        return C_p - C_m


def calc_C(cl: npt.NDArray,
           lmin: int = 2,
           lmax: int | None = None) -> npt.NDArray:
    '''
    Calculate odd/even averaged multipoles.
    Equation (19) in Shi et al. 2023.

    Parameters
    ----------
    cl : array
        The angular power spectrum, default to start with lmin, end with lmax.
    lmin, lmax : int
        The minimum/maximum of the summation.
        Including lmax.
    '''
    ell = np.arange(lmin, lmax + 1, 2)
    factor = (ell * (ell + 1)) / (2 * np.pi)
    length = len(ell)
    C = 1. / length * np.sum(factor * cl, axis=1)
    return C


#####################################
### Hemispherical power asymmetry ###
#####################################

def calc_lvm(m: npt.NDArray,
             mask: npt.NDArray,
             nside_lvm: int = 16,
             rdisc: float = 4.,
             thres: float = 0.9) -> npt.NDArray:
    '''
    Construct local-variance map(s).
    Section 3.4 in Shi et al. 2023.

    Parameters
    ----------
    m : array
        The input map(s), should have dipole removed beforehand.
        If 2D array, the first axis spans over different maps.
    mask : boolean array
        The mask to apply to the maps.
        True stands for pixels to be masked.
        `mask` should have the same dimension as the `m`, with only exception that
        the same mask is applied to all maps. In that case, m is in 2d and mask is
        in 1d.
    nside_lvm : int
        The resolution of the local-variance map.
    rdisc : float
        Radius of the disc, unit: deg.
    thres : float
        Threshold, seen explanation in Section 3.4 text.

    Return
    ------
       Local-variance map(s).
    '''
    m_m = np.atleast_2d(m)
    _mask = np.atleast_2d(mask)
    if len(m_m) != len(_mask):
        _mask = np.repeat(_mask, repeats=len(m_m), axis=0)

    m_m = np.ma.masked_array(data=m_m, mask=_mask)

    nside = hp.get_nside(m_m[0])
    if nside < nside_lvm:
        raise Exception('Cannot make LVM at a higher resolution.')

    vecs = np.array(
        hp.pix2vec(
            nside_lvm, np.arange(
                hp.nside2npix(nside_lvm)))).transpose()

    lvm = np.zeros((len(m_m), len(vecs)))
    for i in range(len(vecs)):
        indx_pix = hp.query_disc(nside, vecs[i], np.deg2rad(rdisc))
        indx_map = np.argwhere(
            (np.sum(_mask[:, indx_pix], axis=1) / len(indx_pix)) < thres).ravel()
        lvm[indx_map, i] = np.ma.var(m_m[indx_map][:, indx_pix], axis=1)
    return lvm


def f_min(params: npt.NDArray,
          lvm: npt.NDArray,
          bias: npt.NDArray = None,
          varlvm: npt.NDArray = None) -> float:
    '''
    Dipole fitting likelihood.
    Equation (20) in Shi et al. 2023.

    Parameters
    ----------
    params : array
        Monopole and dipole parameters, 4 numbers.
    lvm : array
        One LVM.
    bias : array
        The bias of the local-variance map due to the noise.
        If None, will be set to zero.
    varlvm : array
        The variance of different realizations of LVMs.
        If None, will be set to ones.
        zeros are allowed, in which case the corresponding pixels will be ignored.

    Return
    ------
    The chi2 value.
    '''
    if bias is None:
        bias = np.zeros_like(lvm)
    if varlvm is None:
        varlvm = np.ones_like(lvm)

    params = np.array(params)
    m = np.where(varlvm == 0, 0, 1).astype('bool')
    x, y, z = hp.pix2vec(16, np.arange(hp.nside2npix(16)))
    model = np.dot(np.array(params[:, np.newaxis]).T,
                   np.array([np.ones_like(x), x, y, z]))[0]
    return np.sum((lvm[m] - bias[m] - model[m])**2 / varlvm[m])


def fit_lvm_dipole(lvm: npt.NDArray,
                   bias: npt.NDArray = None,
                   varlvm: npt.NDArray = None,
                   method: str = 'Powell') -> npt.NDArray:
    '''
    The fitting routine.

    Parameters
    ----------
    lvm : array
        Local-variance map.
    bias : array
        The bias of the local-variance map due to the noise.
        If None, will be set to zero.
    varlvm : array
        The variance of different realizations of local-variance maps, should be 2D with.
        the first axis spanning over different maps.
        If None, will be set to ones.
    method : str
        The optimization method to use, default is 'Powell'.
        See scipy.optimize.minimize for more options.

    Return
    ------
    The fitted parameters, 4 numbers: [monopole, dipole_x, dipole_y, dipole_z].
    '''
    if bias is None:
        bias = np.zeros_like(lvm)
    if varlvm is None:
        varlvm = np.ones_like(lvm)
    ret = sop.minimize(f_min, x0=[1., 1., 1., 1.],
                       args=(lvm, bias, varlvm), method=method)
    return ret.x
