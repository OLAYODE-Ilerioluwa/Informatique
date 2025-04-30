# Imports

import PIL
import itertools
from PIL import Image
import numpy as np
import scipy as sp
import os
from math import log10, sqrt
from math import cos, pi
from copy import deepcopy
from typing import List
import math
from scipy.fftpack import idct


def load(filename):
    toLoad = Image.open(filename)
    return np.asarray(toLoad)


test = load("test.png")
test2 =Image.open("test.png")


def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def dct2(a):
    return sp.fft.dct(sp.fft.dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')


def idct2(a):
    return sp.fft.idct(sp.fft.idct(a, axis=0, norm='ortho'), axis=1, norm='ortho')

# Question 1


def Ycbcr(matrix):
    # Creates an empty matrix
    matrix2 = np.empty((matrix.shape[0], matrix.shape[1], 3))

    for i in range(matrix2.shape[0]):
        for j in range(matrix2.shape[1]):

            # Calculates the Y, Cb, and Cr values
            Y = int(0.299 * matrix[i, j, 0] + 0.587 *
                    matrix[i, j, 1] + 0.114 * matrix[i, j, 2])
            Cb = int(-0.1687 * matrix[i, j, 0] - 0.3313 *
                     matrix[i, j, 1] + 0.5 * matrix[i, j, 2] + 128)
            Cr = int(0.5 * matrix[i, j, 0] - 0.4187 *
                     matrix[i, j, 1] - 0.0813 * matrix[i, j, 2] + 128)

            # Assigns the YCbCr values to the corresponding position in the new matrix
            matrix2[i, j] = (Y, Cb, Cr)

    # Prints the YCbCr values of the first pixel in the new matrix
    print(matrix2[0, 0])

    return matrix2

# Question 2


def RGb(matrix):
    # Creates an empty matrix, unit8 allows us to use 8bit values, meaning ranging from 0 to 255
    matrix2 = np.empty((matrix.shape[0], matrix.shape[1], 3), dtype=np.uint8)

    for i in range(matrix2.shape[0]):
        for j in range(matrix2.shape[1]):

            # Calculates the RGB values
            R = matrix[i, j, 0] + 1.402 * (matrix[i, j, 2] - 128)
            V = matrix[i, j, 0] - 0.34414 * \
                (matrix[i, j, 1] - 128) - 0.71414 * (matrix[i, j, 2] - 128)
            B = matrix[i, j, 0] + 1.772 * (matrix[i, j, 1] - 128)

            # Assigns the RGB values
            matrix2[i, j] = (np.uint8(np.clip(R, 0.0, 255.0)), np.uint8(
                np.clip(V, 0.0, 255.0)), np.uint8(np.clip(B, 0.0, 255.0)))

    return matrix2

# Question 3


def pading_verif(matrice):
    
    width, height = matrice.size
    print(width, height)

    # Adjusts the width and height to the nearest multiple of 8
    while width % 8 != 0:
        width = width + 1
    while height % 8 != 0:
        height = height + 1

    # Creates a new image with the adjusted width and height, gets filled with black around
    result = Image.new(matrice.mode, (width, height), (0, 0, 0))

    # Pastes the original image to the new one
    result.paste(matrice, (0, 0))

    result.save("test3.jpg")

    return result


def supp_pading(matrix, matrix2):

    pad_width, pad_height = matrix.size
    width, height = matrix2.size

    # Prints the original padding width and height
    print(pad_width, pad_height)

    # Prints the new width and height
    print(width, height)

    # Checks if the padding width needs an adjustment
    if pad_width != width:
        pad_width = width
    else:
        None

    # Checks for the height
    if pad_height != height:
        pad_height = height
    else:
        None

    # Creates a new image with the adjusted padding width and height and gets filled with black
    result = Image.new(matrix.mode, (pad_width, pad_height), (0, 0, 0))

    result.paste(matrix, (0, 0))
    result.save("test3.jpg")


# Question 4

def samp(mat):

    # Creates an empty matrix 3x3
    # Devides the second dimension by 2,the 3 is for the RGB values
    matI = np.empty((mat.shape[0], mat.shape[1]//2, 3), dtype=np.uint8)

    for i in range(matI.shape[0]):
        if i == 471:
            break
        for j in range(matI.shape[1]):

            # Extracts the Red value from the current pixel of mat
            R = mat[i][2*j][0]

            # Calculates the average of the Green values (current + next pixel)
            G = (mat[i][2*j][1] + mat[i][2*j+1][1]) // 2

            # Calculates the average of the Blue values (current + next pixel)
            B = (mat[i][2*j][2] + mat[i][2*j+1][2]) // 2

            # Assigns the calculated RGB values to the corresponding pixel in matI
            matI[i][j] = (R, G, B)

    return matI

# Question 5


def reverse_samp(matI):
    # Gets the height and width
    h, w2, c = matI.shape

    # Doubles the width
    w = w2 * 2

    # Creates an empty matrix with the same height but doubled width
    mat = np.empty((h, w, c), dtype=np.uint8)

    # Fills the even columns of the matrix with the corresponding columns of the initial matrix
    mat[:, ::2, :] = matI

    # Repeats each column of the first matrix twice horizontally, and then fills the odd columns of the output matrix with the corresponding repeated columns
    mat[:, 1::2, :] = np.repeat(matI, 2, axis=1)[:, 1::2, :]

    return mat


# Question 6

def bloc(old_table: List[List[int]]) -> List[List[int]]:
    """
    Applies the Discrete Cosine Transform (DCT) to the input old_table.

    Args:
        old_table (List[List[int]]): The original table to be transformed.

    Returns:
        List[List[int]]: The transformed table after applying the DCT.
    """

    pi = math.pi  # Assigns the value of pi
    # Creates a new table 'new_table' by copying 'old_table'
    new_table = deepcopy(old_table)

    for new_y in range(8):
        # Calculates the cos of an angle and assigns it to the variable 'cy'
        cy = cos(((2 * new_y + 1) * pi) / 16)

        for new_x in range(8):
            # Calculates the cos of an angle and assigns it to the variable 'cx'
            cx = cos(((2 * new_x + 1) * pi) / 16)

            # Calculates the new value of an element in the list using the formula for the DCT
            new_value = sum(
                old_table[old_y][old_x] * cos(((2 * old_y + 1) * new_y * pi) / 16) * cos(
                    ((2 * old_x + 1) * new_x * pi) / 16)
                for old_y in range(8) for old_x in range(8)
            )

            if new_y == 0:
                # Divides the new values by the square root of 2 for the first row
                new_value /= sqrt(2)
            if new_x == 0:
                # Divides the new values by the square root of 2 for the first column
                new_value /= sqrt(2)

            # Multiplies the new value by (0.25 * cx * cy)
            new_value *= (0.25 * cx * cy)
            # Updates the new table with the new value
            new_table[new_y][new_x] = new_value

    return new_table

# Question 7


def transform(blocks):

    # Applies the DCT to the blocks
    blocks = dct2(blocks)

    # Creates a new array bloc (copy of blocks but type set to int)
    bloc = np.array(blocks, dtype=int)

    return bloc


def transform2(blocks):

    # Applies the IDCT to the blocks
    blocks = idct2(blocks)

    # Creates a new array bloc (copy of blocks but type set to int)
    bloc = np.array(blocks, dtype=int)
    return bloc


# Question 8

def threshold_coef(seuil, bloc):

    for i in range(bloc.shape[0]):
        for j in range(bloc.shape[1]):
            for w in range(bloc.shape[2]):

                # Checks if the value at bloc[i][j][w] is below the threshold
                if seuil > bloc[i][j][w]:

                    # Sets the value to 0 if it is below the threshold
                    bloc[i][j][w] = 0

    return bloc

# Question 9


def multi_mode(mode, image):
    # Checks the mode value to determine the operation to perform
    if mode == 0:
        # Applies the transform operation to the image
        print(transform(bloc(image)))

    elif mode == 1:
        # Applies the transform operation to the image and applies thresholding with threshold 3
        print(threshold_coef(3, transform(bloc(image))))

    elif mode == 2:
        # Applies the transform operation to the image after applying channel transformation and applies thresholding with threshold 3
        print(threshold_coef(3, transform(bloc(samp(image)))))


# Question 10 and 11

def writing(image, mode, encoding):
    # Defines the dimensions of the image
    test = load("test.png")
    height = test.shape[0]
    width = test.shape[1]
    test2 = transform(bloc(test))

    # Defines the compression mode and encoding type
    Y = ""
    Cr = ""
    Cb = ""

    # Writes the information in a text file
    with open("text.txt", "w") as f:
        f.write("SJPG\n")  # Header which indicates the file format
        # Writes the height and width of the image
        f.write(f"{height} {width}\n")
        f.write(f"mode: {str(mode)} \n")  # Writse the compression mode

        if encoding == 1:
            # Writes the encoding type as RLE if encoding is 1
            f.write(f"RLE \n")
        else:
            # Writes the encoding type as NORLE if it's not 1
            f.write(f"NORLE \n")

        # Writes the Y channel values to the file
        for i in range(height):
            for j in range(width):
                # Appends the Y channel value to Y string
                Y += str(test2[i][j][0]) + " "
        f.write(f"{str(Y)}\n")  # Writes Y string to the file

        if encoding == 1:
            rle(Y)  # Applies the RLE encoding to the Y string if the encoding is 1

        # Writes the Cb channel values to the file
        for i in range(height):
            for j in range(width):
                # Appends the Cb channel value to the Cb string
                Cb += str(test2[i][j][1]) + " "
        f.write(f"{str(Cb)}\n")  # Write the Cb string to the file

        if encoding == 1:
            rle(Cb)  # Applies RLE encoding to the Cb string if encoding is 1

        # Writes the Cr channel values to the file
        for i in range(height):
            for j in range(width):
                # Appends the Cr channel value to the Cr string
                Cr += str(test2[i][j][2]) + " "
        f.write(f"{str(Cr)}")  # Writes the Cr string to the file

        if encoding == 1:
            rle(Cr)  # Applies the RLE encoding to the Cr string if the encoding is 1

# Question 12


def rle(line):
    encoding = []  # List to store the RLE encoding
    counter = 1  # Counter to keep track of consecutive zeros

    for i in range(1, len(line)):
        # Checks if the current element is the same as the previous element and if both are zeros
        if line[i] == line[i-1] == 0:
            counter += 1  # Increments the counter if consecutive zeros are found
        else:

            # Appends the RLE encoding of the previous sequence of zeros and its count to the encoding list
            encoding.append(f"{counter}x{line[i-1]}")
            counter = 1  # Resets the counter for the new sequence of zeros

    # Appends the RLE encoding of the last sequence of zeros and its count to the encoding list
    encoding.append(f"{counter}x{line[-1]}")

    return encoding


# Question 13

def decompress_blocks(blocks):

    # Creates empty arrays to store the YCBCR values
    Y = np.zeros((blocks[0][0].shape[0] * len(blocks),
                 blocks[0][0].shape[1] * len(blocks[0])))
    Cb = np.zeros((blocks[1][0].shape[0] * len(blocks),
                  blocks[1][0].shape[1] * len(blocks[0])))
    Cr = np.zeros((blocks[2][0].shape[0] * len(blocks),
                  blocks[2][0].shape[1] * len(blocks[0])))

    # Iterates over the blocks and performs inverse discrete cosine transform (IDCT)
    for i in range(len(blocks)):
        for j in range(len(blocks[0])):

            # Applies IDCT to each block of YCBCR values
            Y_block = idct2(idct2(blocks[0][i][j], norm='ortho'), norm='ortho')
            Cb_block = idct2(
                idct2(blocks[1][i][j], norm='ortho'), norm='ortho')
            Cr_block = idct2(
                idct2(blocks[2][i][j], norm='ortho'), norm='ortho')

            # Assigns the IDCT transformed blocks to Y CB and CR
            Y[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = Y_block
            Cb[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = Cb_block
            Cr[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = Cr_block

    # Converts to RGB
    r = Y + 1.402 * (Cr - 128)
    g = Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)
    b = Y + 1.772 * (Cb - 128)

    # Creates an empty image with the Y channel shape
    image = np.zeros((Y.shape[0], Y.shape[1], 3))

    # Assigns the RGB to the image
    image[:, :, 0] = r
    image[:, :, 1] = g
    image[:, :, 2] = b

    # Clips the image values to the range [0, 255] and converts the data type to 8-bit
    return np.clip(image, 0, 255).astype('uint8')

# Question 14


def read_sjpg(file_name):
    # Reads the content of the file
    with open(file_name, 'r') as f:
        content = f.readlines()

    # Checks if the file starts with 'SJPG'
    if content[0] != 'SJPG\n':
        raise ValueError("The file must start with 'SJPG'")

    blocks = []  # List to store the blocks

    block_index = 1  # Index to keep track of the current block

    while block_index < len(content):
        # Gets the number of values in the block
        num_values_block = int(content[block_index])

        # Extracts the values from the content and converts them to integers
        values_bloc = [int(value) for value in content[block_index +
                                                       1: block_index + num_values_block + 1]]

        # Appends the block to the blocks list
        blocks.append(values_bloc)

        # Moves the block index to the next block
        block_index += num_values_block + 1

    return blocks



# print(writing(multi_mode(1,test),1))
# psnr(test,RGb(Ycbcr(test)))
# Image.fromarray(test,'RGB').show()
# print(paDingverif(test2))
# print(suppPadin(test3,test2))
# Image.fromarray(samp(test),'RGB').show()
# Image.fromarray(reverse_samp(samp(test)),'RGB').show()
# print(transform(blocs(test)))
# print(transform(bloc(test)))
# print(apply_dct(bloc(test)))
# print(transormée(bloc))
# print(writing(test2,"text.txt"))
# print(bloc(test))
# print(psnr(test,RGb(Ycbcr(test))))
# print(threshold_coef(0, dct2(bloc(test))))
# print(threshold_coef(3,transform(bloc(test))))
# print(multi_mode(2,test))
