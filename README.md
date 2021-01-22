# Testing Images and Datasets for "Benfordness"
### Images
Quantized discrete cosine transform (DCT) coefficients from a natural image should follow Benford's Law. To test this, I extracted the first digits of the five lowest frequency DCT coefficients (calculated from every 8x8 pixel area within the grayscale image), and plotted their frequency aganist Benford's Law.

1. Transform image to grayscale
2. Calculate DCT coefficients from N 8x8 pixel blocks
3. Quantize coefficients using Δ90
4. Perform zigzag scan and select the five lowest frequency coefficients
5. Count the first digit of these coefficients for all N blocks


### Dataset Matrix
Out of curosity, I tested coverage diagrams for Benfordness by plotting the frequency of first digits in propagation factor measurements and electric potential measurements, seperately. The later followed Benford's Law.





