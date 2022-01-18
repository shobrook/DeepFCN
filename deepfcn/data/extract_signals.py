from nilearn import image


def extract_signals(niimg, roi_masker):
    niimg = image.load_img(niimg) if isinstance(niimg, str) else niimg
    confounds = image.high_variance_confounds(niimg)
    time_series = masker.fit_transform(niimg, confounds=confounds)

    return time_series
