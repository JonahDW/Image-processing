
def write_fits_mask(imname, image_file, thresh_isl, rms_image=None):
    """
    Write an output fits containing binary mask

    Keyword arguments:
    outfile -- Name of the output mask file (CRTF)
    regions -- Region or list of regions to write
    size -- Multiply input major and minor axes by this amount
    """
    image = helpers.open_fits_casa(image_file)
    if rms_image is None:
        rms_image = imname+'_rms.fits'
    rms = helpers.open_fits_casa(rms_image)

    snr = np.squeeze(image[0].data)/np.squeeze(rms[0].data)
    mask = np.zeros(snr.shape)
    mask[snr > thresh_isl] = 1

    # Store in existing HDU and write to fits
    rms[0].data[0,0,:,:] = mask

    outfile = imname+'_mask.fits'
    print(f'Wrote mask file to {outfile}')
    rms.writeto(outfile, overwrite=True)