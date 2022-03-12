from nilearn import image


def extract_signals(niimg, roi_masker):
    """
    Extracts BOLD signals (e.g. time-series) from each ROI in an fMRI scan (e.g.
    Niimg).

    Parameters
    ----------
    niimg : Niimg-like object, string
        fMRI scan to extract signals from; if string, represents a path to a
        NIfTI image
    roi_masker : NiftiMasker object
        ROI mask that tells nilearn which ROIs to extract signals from

    Returns
    -------
    numpy.ndarray
        Array of BOLD signals; shape = [num_rois, time_series_len]
    """
    niimg = image.load_img(niimg) if isinstance(niimg, str) else niimg
    confounds = image.high_variance_confounds(niimg)
    time_series = roi_masker.fit_transform(niimg, confounds=confounds)

    return time_series
