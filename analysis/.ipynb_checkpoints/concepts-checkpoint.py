import warnings
import os
import numpy as np
import pandas as pd
import skimage.io
import matplotlib.pyplot as plt

from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings('ignore')


def pixel_color_checker(data, row_index, pixel_index, R, G, B):
    """
    :param data: image data as array
    :param row_index: index of image row to be checked
    :param pixel_index: index of pixel to be checked
    :param R: value (in range [0, 255]) at the R (red) value of the RGB notation
    :param G: value (in range [0, 255]) at the G (green) value of the RGB notation
    :param B: value (in range [0, 255]) at the B (blue) value of the RGB notation
    :return: boolean specifying whether the pixel of interest is colored as specified by the RGB values
    """
    if (data[row_index][pixel_index][0] == R) and (data[row_index][pixel_index][1] == G) \
            and (data[row_index][pixel_index][2] == B):
        return True
    else:
        return False


def pairs_stem(data):
    """
    :param data: image data as array
    :return: - integer list (integers in [0, 3]) of length that of the pre-miRNA, the integers correspond to the pairing
                types in the stem (0: gap or mismatch, 1: CG/GC, 2: AU/UA, 3: GU/UG wobble)
             - pixel index denoting the start of the pre-miRNA stem (going from right to left).
    """
    # find the first stem pair (= first colored pixel in the middle image row and going from right to left)
    first_pair = 0
    row_before_middle = 12
    for pixel_index in range(99, -1, -1):
        # in case of a white pixel, continue to the next pixel (the one to the left)
        if pixel_color_checker(data, row_before_middle, pixel_index, 255, 255, 255):
            pixel_index += 1
        else:
            # if not a white pixel, we have reached the first colored pixel from right to left
            first_pair = pixel_index
            break

    # initialize an empty list where all the pairing integers will be stored
    pairs = []
    # iterate over pixel (bars) from right to left starting at the stem begin and check the color as well as
    # the length of a pixel bar, since these two things define the type of nucleotide pair
    for pixel_index in range(first_pair, -1, -1):
        # C-G pair (blue-red with bar length 2 for both bars)
        if pixel_color_checker(data, row_before_middle - 1, pixel_index, 0, 0, 255) and \
                pixel_color_checker(data, row_before_middle - 2, pixel_index, 255, 255, 255) and \
                pixel_color_checker(data, row_before_middle + 2, pixel_index, 255, 0, 0) and \
                pixel_color_checker(data, row_before_middle + 3, pixel_index, 255, 255, 255):
            pairs.append(1)  # CG pairing type is referred to as 1
        # G-C pair (red-blue with bar length 2 for both bars)
        elif pixel_color_checker(data, row_before_middle - 1, pixel_index, 255, 0, 0) and \
                pixel_color_checker(data, row_before_middle - 2, pixel_index, 255, 255, 255) and \
                pixel_color_checker(data, row_before_middle + 2, pixel_index, 0, 0, 255) and \
                pixel_color_checker(data, row_before_middle + 3, pixel_index, 255, 255, 255):
            pairs.append(1)  # GC pairing type is referred to as 1
        # U-A (green-yellow with bar length 3 for both)
        elif pixel_color_checker(data, row_before_middle - 2, pixel_index, 0, 255, 0) and \
                pixel_color_checker(data, row_before_middle - 3, pixel_index, 255, 255, 255) and \
                pixel_color_checker(data, row_before_middle + 3, pixel_index, 255, 255, 0) and \
                pixel_color_checker(data, row_before_middle + 4, pixel_index, 255, 255, 255):
            pairs.append(2)  # UA pairing type is referred to as 2
        # A-U (yellow-green with bars of length 3)
        elif pixel_color_checker(data, row_before_middle - 2, pixel_index, 255, 255, 0) and \
                pixel_color_checker(data, row_before_middle - 3, pixel_index, 255, 255, 255) and \
                pixel_color_checker(data, row_before_middle + 3, pixel_index, 0, 255, 0) and \
                pixel_color_checker(data, row_before_middle + 4, pixel_index, 255, 255, 255):
            pairs.append(2)  # AU pairing type is referred to as 2
        # G-U wobble (red-green with length 4 for both bars)
        elif pixel_color_checker(data, row_before_middle - 3, pixel_index, 255, 0, 0) and \
                pixel_color_checker(data, row_before_middle - 4, pixel_index, 255, 255, 255) and \
                pixel_color_checker(data, row_before_middle + 4, pixel_index, 0, 255, 0) and \
                pixel_color_checker(data, row_before_middle + 5, pixel_index, 255, 255, 255):
            pairs.append(3)  # GU pairing type is referred to as 3
        # U-G wobble (green-red with length 4 for both bars)
        elif pixel_color_checker(data, row_before_middle - 3, pixel_index, 0, 255, 0) and \
                pixel_color_checker(data, row_before_middle - 4, pixel_index, 255, 255, 255) and \
                pixel_color_checker(data, row_before_middle + 4, pixel_index, 255, 0, 0) and \
                pixel_color_checker(data, row_before_middle + 5, pixel_index, 255, 255, 255):
            pairs.append(3)  # UG pairing type is referred to as 3
        else:
            pairs.append(0)  # all remaining pairs (gaps and mismatches) are referred to as 0

    return pairs, first_pair


def loop_concepts(data, pairs):
    """
    :param data: image data as array
    :param pairs: list of references to types of pairs in the sequence stem
    :return: concepts related to the terminal loop: - presence of the loop
                                                    - starting pixel and row (counting from left to right, and in
                                                            the upper half of the image)
                                                    - pixel and row of highest point (from bottom to top) of the loop
                                                    - the length and width of the loop
    """

    # Terminal loop presence: is there a gap in the first pair of the pre-miRNA (from left to right) in either the upper
    # or lower half. This can be checked by going over the 12th and 13th row of the image and check for a black pixel
    # as first pixel in one of these rows.
    row_before_middle = 12
    loop_present = None
    row_gap = None
    for row_index in range(row_before_middle, row_before_middle + 2):
        if pixel_color_checker(data, row_index, 0, 0, 0, 0):
            loop_present = False
            row_gap = row_index
            break
        else:
            loop_present = True

    # If present, estimate the terminal loop width and length by estimating the starting and end point (row, pixel)
    # by finding the first occurrence of a pair that is not a gap or mismatch (i.e., 0) in the pairs list
    # the pairs list is ordered from right to left, so first reverse the order
    pairs_reversed = pairs[::-1]
    if loop_present:
        # step 1: find lowest point in loop (i.e., the starting pixel and row (going from right to left))
        # the first non-zero element in the pairs list is the start of the loop
        try:
            loop_start_pixel = np.nonzero(np.array(pairs_reversed))[0][0] - 1  # - 1 since we start counting from pixel 0
        except:
            loop_start_pixel = 4
        # step 2: find highest point in loop (in the upper half of the image) (= first colored pixel going from top to
        # bottom in rows and from left to right in pixels)
        loop_highest_pixel = loop_start_pixel  # initial value for the pixel: starting point of the loop
        loop_highest_row = 12  # initial value for the row: the row just before the middle of the image is reached
        for row_index in range(0, row_before_middle):
            for pixel_index in range(0, loop_start_pixel):
                if pixel_color_checker(data, row_index, pixel_index, 255, 255, 255):
                    continue
                elif ((pixel_index < loop_highest_pixel) and (row_index < loop_highest_row)) or \
                        ((pixel_index > loop_highest_pixel) and (row_index == loop_highest_row)):
                    loop_highest_pixel = pixel_index
                    loop_highest_row = row_index

        # step 3: width of the loop
        loop_width = loop_start_pixel + 1

        # step 4: get the lowest point of the loop by going over all rows in the bottom half of the image and the pixel
        # index of the highest point in the loop
        lowest_row = 24
        for row_index in range(24, row_before_middle, -1):
            if pixel_color_checker(data, row_index, loop_highest_pixel, 255, 255, 255):
                continue
            else:
                lowest_row = row_index + 1  # note that this is the row below the lowest row
                break

        # step 5: length of the loop (note that the loop is symmetrical!)
        # Case 1: lowest point is the last image row and the highest point the first image row
        # --> the loop length is equal to the image length (i.e., 25)
        if (lowest_row == 24) and (loop_highest_row == 0):
            loop_length = 25
        else:
            # Case 2: loop does not reach the image borders
            loop_length = 25 - ((25 - lowest_row) + loop_highest_row)

        # finally, the concept that is linked to the case when there is no loop present should be empty in case there
        # is a loop
        width_gap = np.nan

    else:
        # in case there is no loop present, measure the width of the gap (# of black pixels)
        width_gap = 0
        for pixel_index in range(0, 100):
            if pixel_color_checker(data, row_gap, pixel_index, 0, 0, 0):
                width_gap += 1
            else:
                break
        # set all other concepts related to the loop as empty valued variables
        width_gap = width_gap
        loop_start_pixel = np.nan
        loop_highest_row = np.nan
        loop_highest_pixel = np.nan
        loop_length = np.nan
        loop_width = np.nan

    return loop_present, loop_start_pixel, loop_highest_row, loop_highest_pixel, loop_length, loop_width, width_gap


def ugu_motif(data, loop_present, loop_highest_pixel, loop_start_pixel):
    """
    :param data: image data as array
    :param loop_present: boolean whether terminal loop is present or not
    :param loop_highest_pixel: highest point of the terminal loop
    :param loop_start_pixel: starting point of the terminal loop
    :return: boolean specifying whether the sequence contains the motif
    """
    # for the motif to be present, there needs to be a terminal loop
    motif_present = False
    if loop_present:
        # check whether the UGU-motif is either in the loop (from the highest point to the start) or just
        # before the loop (i.e., 4 pixels (nt) after the loop) in the upper image half
        row_before_middle = 12
        for pixel in range(loop_highest_pixel, loop_start_pixel + 4):
            if pixel_color_checker(data, row_before_middle, pixel, 255, 0, 0):
                if pixel_color_checker(data, row_before_middle, pixel + 1, 0, 255, 0):
                    if pixel_color_checker(data, row_before_middle, pixel + 2, 255, 0, 0):
                        motif_present = True
                    else:
                        continue
        for pixel in range(loop_highest_pixel, loop_start_pixel + 4):
            if pixel_color_checker(data, row_before_middle, pixel, 0, 255, 0):
                if pixel_color_checker(data, row_before_middle, pixel + 1, 255, 0, 0):
                    if pixel_color_checker(data, row_before_middle, pixel + 2, 0, 255, 0):
                        motif_present = True
                    else:
                        continue

    return motif_present


def AU_pairs_begin_maturemiRNA(pairs):
    """
    :param pairs: list of references to types of pairs in stem of pre-miRNA
    :return: boolean whether a successions of at least 2 AU/UA pairs was found in the area around 18-25 nt from the
    stem begin
    """
    # look for a succession/presence of 2+ AU/UA pairs in the area around 18-25 nt from the stem begin
    area_of_interest = pairs[18:26]
    AU_motif = any((area_of_interest[i], area_of_interest[i + 1]) == (2, 2) for i in range(len(area_of_interest) - 1))

    # if the length of the pre-miRNA < 19, it cannot be a pre-miRNA so there also cannot be a AU motif
    if len(area_of_interest) < 2:
        AU_motif = False

    return AU_motif


def bulge_info(bulges_list, data):
    """
    :param bulges_list: list containing indices of all bulges in the pre-miRNA
    :param data: image data as array
    :return: information on the bulges such as width, height, shape, start and end point, nucleotide types
    """
    row_before_middle = 12

    # go over the bulges in the list of bulges add generate the information
    bulge_info_list = []
    for bulge in bulges_list:
        bulge_width = len(bulge)  # the width of the bulge is equal to the total length of the bulge

        # find the highest point (row,pixel) of the bulge by going over the rows in the upper half of the image and the
        # pixels that are part of the bulge to find which pixel is located highest and has a white pixel above it (or
        # the pixel is located in the first row and there is nothing above it)
        bulge_highest_pixel = 100
        bulge_highest_row = row_before_middle
        for row_index in range(0, row_before_middle):
            # find first occurrence of colored pixels and store this index
            for pixel_index in range(bulge[0], bulge[-1] + 1):
                if pixel_color_checker(data, row_index, pixel_index, 255, 255, 255):
                    continue
                else:
                    if ((pixel_index < bulge_highest_pixel) and (row_index < bulge_highest_row)) or \
                            ((pixel_index > bulge_highest_pixel) and (row_index == bulge_highest_row)):
                        bulge_highest_pixel = pixel_index
                        bulge_highest_row = row_index

        # similar for the lower half of the image to find the lowest point of the bulge
        bulge_lowest_row = 24
        bulge_lowest_pixel = None
        # loop over the rows from lowest to highest
        for row_index in range(24, row_before_middle, -1):
            for pixel_index in range(bulge[0], bulge[-1] + 1):
                if pixel_color_checker(data, row_index, pixel_index, 255, 255, 255):
                    continue
                else:
                    bulge_lowest_row = row_index + 1  # +1 bc we are dealing with index range starting at 0
                    bulge_lowest_pixel = pixel_index
                    break

        # in case the bulge's highest and lowest point are the first and last row, the length is equal to 25
        if (bulge_lowest_row == 25) and (bulge_highest_row == 0):
            bulge_length = 25
        # in case the lowest point is the lowest row, the length is equal to 25 - highest point
        elif (bulge_lowest_row == 25) and (bulge_highest_row != 0):
            bulge_length = 25 - bulge_highest_row
        else:
            bulge_length = 25 - ((25 - bulge_lowest_row) + bulge_highest_row)

        # get color of pairs by checking the pixel RGB values
        nucleotide_pairs = []
        colors = []
        for pixel in range(bulge[0], bulge[-1] + 1):
            pair = []
            color = []
            for row in range(row_before_middle, row_before_middle + 2):
                if pixel_color_checker(data, row, pixel, 0, 0, 0):
                    pair.append('gap')
                    color.append('black')
                elif pixel_color_checker(data, row, pixel, 255, 0, 0):
                    pair.append('G')
                    color.append('red')
                elif pixel_color_checker(data, row, pixel, 0, 255, 0):
                    pair.append('U')
                    color.append('green')
                elif pixel_color_checker(data, row, pixel, 0, 0, 255):
                    pair.append('C')
                    color.append('blue')
                elif pixel_color_checker(data, row, pixel, 255, 255, 0):
                    pair.append('A')
                    color.append('yellow')
            # combine the two nucleotides so that the pair is saved as ..-..
            nucleotide_pairs.append(pair[0] + "-" + pair[1])
            colors.append(color[0] + '-' + color[1])

        # find the most occurring color in the bulge by computing the counts and finding the max count
        values, counts = np.unique(colors, return_counts=True)
        highest_count_index = np.argmax(counts)
        most_occurring_color = values[highest_count_index]

        # add all the generated info into a list and add this list to an overview list
        bulge_info_list.append([bulge, (bulge_highest_row, bulge_highest_pixel), (bulge_lowest_row, bulge_lowest_pixel),
                                bulge_width, bulge_length, nucleotide_pairs, colors, most_occurring_color])

    return bulge_info_list


def bulges(data, pairs, loop_start_pixel, begin):
    """
    :param data: image data as array
    :param pairs: list of references to pairs inside pre-miRNA
    :param loop_start_pixel: pixel index of starting point of loop (from right to left)
    :param begin: beginning index of stem (from right to left)
    :return: list of symmetric and asymmetric bulges, as well as more detailed list containing information on the size,
    height, nucleotide pairs, shape, etc. of the bulge
    """
    row_before_middle = 12

    # we look for bulges in the area after the loop, hence we need to slice the pairs list from the stem begin
    # until the start point of the loop. Recall that the pairs list is ordered from right to left.
    if np.isnan(loop_start_pixel):
        loop_start_pixel = -1
    pairs_until_loop = pairs[0:len(pairs) - int(loop_start_pixel) - 1]
    # -1 bc we should stop collecting pairs one before the loop starts, otherwise pairs from the loop are also added

    # get all the indices where the pair is part of a bulge (i.e., 0). Since the base_pairs list is in reverse order,
    # we do stem begin - index of the list item
    bulge_indices = [begin - i for i, e in enumerate(pairs_until_loop) if e == 0]

    # create lists of lists from items bulge_indices so that successive indices are in one list
    bulge_indices_grouped = []
    bulge_list = []  # initialize an empty list where the indices of one bulge will be stored
    for i in range(len(bulge_indices) - 1):
        # Case 1: bulge contains only 1 pair
        if (bulge_indices[i] + 1 != bulge_indices[i - 1]) and (bulge_indices[i] - 1 != bulge_indices[i + 1]):
            single_bulge = [bulge_indices[i]]
            bulge_indices_grouped.append(single_bulge)
        else:
            # Case 2: bulge consists of multiple pairs
            # first, check whether the succeeding indices only differ by 1
            if bulge_indices[i] - 1 == bulge_indices[i + 1]:
                bulge_list.append(bulge_indices[i])
                # if the index is equal to the final index in the range of the for-loop, add the next index to the
                # bulge list (this is the final index of the last bulge)
                if i == (len(bulge_indices) - 2):
                    bulge_list.append(bulge_indices[i + 1])
                    bulge_indices_grouped.append(bulge_list)
            else:
                # if the difference between the succeeding indices is larger then 1, we have reached the final index of
                # a bulge
                bulge_list.append(bulge_indices[i])
                bulge_indices_grouped.append(bulge_list)
                bulge_list = []
                # special case: the final item in the bulge indices is a single bulge
    if len(bulge_indices) > 1:
        if bulge_indices[-1] != bulge_indices[-2] - 1:
            single_bulge = [bulge_indices[-1]]
            bulge_indices_grouped.append(single_bulge)
    elif len(bulge_indices) == 1:
        single_bulge = bulge_indices
        bulge_indices_grouped.append(single_bulge)

    symmetric_bulges = []
    asymmetric_bulges = []
    # For each bulge, check whether it contains a gap (i.e. asymmetric bulge) or not (i.e. symmetric bulge)
    for bulge_list in bulge_indices_grouped:
        gap_test = None
        for pixel_index in bulge_list:
            if pixel_color_checker(data, row_before_middle, pixel_index, 0, 0, 0):
                gap_test = True
            elif pixel_color_checker(data, row_before_middle + 1, pixel_index, 0, 0, 0):
                gap_test = True
            else:
                gap_test = False
        # if gap_test is true, there is a gap in the bulge and we are dealing with an asymmetric bulge
        # also, reverse the order in the list so that the bulge is ordered from left to right
        if gap_test:
            asymmetric_bulges.append(bulge_list[::-1])
        else:
            symmetric_bulges.append(bulge_list[::-1])

    # reverse the order of the list of bulges to match with the order in the image (from left to right)
    symmetric_bulges = symmetric_bulges[::-1]
    asymmetric_bulges = asymmetric_bulges[::-1]

    # get information on the bulges such as width, height, start and end point, nucleotide types inside bulge
    symmetric_bulge_info = bulge_info(symmetric_bulges, data)
    asymmetric_bulge_info = bulge_info(asymmetric_bulges, data)

    return symmetric_bulges, asymmetric_bulges, symmetric_bulge_info, asymmetric_bulge_info


def palindrome(data):
    """
    :param data: image data as array
    :return: score referring to what extent the structure of the two image halves is symmetrical
    """

    # loop over both lists simultaneously and if the counts are equal, add True, else, add False
    # lastly, divide the amount of Trues by the length of the pre-miRNA to get the symmetrical structure score
    row_before_middle = 12

    # for both images halves, count the number of colored pixels in a bar (i.e., its length) until a white pixel occurs
    upper_half_counts = []
    lower_half_counts = []
    for pixel_index in range(0, 100):
        upper_count = 0
        lower_count = 0
        # check the upper half of the image
        for row_index in range(row_before_middle, -1, -1):
            if pixel_color_checker(data, row_index, pixel_index, 255, 255, 255):
                upper_half_counts.append(upper_count)
                break
            # if the colored pixels go up to the first row, there is no white pixel above that any more so we need
            # to add the current count to the list
            elif row_index == 0:
                upper_half_counts.append(upper_count)
                break
            # if above cases do not hold, we have found a colored pixel and the count should go up
            else:
                upper_count += 1
        # check the lower half of the image
        for row_index in range(row_before_middle + 1, 25):
            if pixel_color_checker(data, row_index, pixel_index, 255, 255, 255):
                lower_half_counts.append(lower_count)
                break
            # if the colored pixels go up to the last row, there is no white pixel below that any more so we need
            # to add the current count to the list
            elif row_index == 24:
                lower_half_counts.append(lower_count + 1)
                break
            # if above cases do not hold, we have found a colored pixel and the count should go up
            else:
                lower_count += 1

    # zip the count lists and go over them to check whether the counts are equal (= symmetric structure) or not
    # also check whether a large asymmetric bulge is included (pixel bar reaching image border) and in which of the
    # two image halves the bulge is located
    palindrome_array = []
    for pixel_upper, pixel_lower in zip(upper_half_counts, lower_half_counts):
        if pixel_upper == pixel_lower:
            palindrome_array.append(True)
        else:
            palindrome_array.append(False)

    # divide the number of True/False by the length of the pre-miRNA (= the array of non-zero counts)
    if 0 in upper_half_counts:
        len_premiRNA = upper_half_counts.index(0)
    else:
        len_premiRNA = len(upper_half_counts)
    subset_palindrome_array = palindrome_array[0:len_premiRNA]
    # get the frequencies for the True and False in the array
    subset_values, subset_counts = np.unique(subset_palindrome_array, return_counts=True)
    # if only True or only False, then the counts is length one --> change counts from [1.] into [0., 1.]
    if all(subset_palindrome_array):
        subset_counts = np.array([0, 1 * len_premiRNA])
    # if only False then change counts from [1.] into [1., 0.]
    elif not any(subset_palindrome_array):
        subset_counts = np.array([1 * len_premiRNA, 0])
    # palindrome score is an array with [(False counts / len_premiRNA), (True counts / len_premiRNA)]
    score = subset_counts / len_premiRNA

    return score, upper_half_counts, lower_half_counts, len_premiRNA


def large_asymmetric_bulge(data):
    """
    :param data: image data as array
    :return: the width and location of the largest asymmetric bulge (if any) in the sequence
    """
    # retrieve the lengths of the bars in the sequences (the counts) from the palindrome function
    score, upper_half_counts, lower_half_counts, len_premiRNA = palindrome(data)

    # zip the count lists and check whether a large asymmetric bulge is included (pixel bar reaching image border)
    # and in which of the two image halves the bulge is located
    bulge_array = []
    bulge_locations = []
    # go over the bar lengths in the counts arrays and check whether they match the large asymmetric bulge requirements
    for pixel_upper, pixel_lower in zip(upper_half_counts[0:len_premiRNA], lower_half_counts[0:len_premiRNA]):
        if pixel_upper == pixel_lower:
            bulge_array.append(0)
            bulge_locations.append(0)
        else:
            # check for large asymmetric bulge in lower half of image
            if pixel_upper == 2 and pixel_lower == 12:
                bulge_array.append(1)
                bulge_locations.append('lower')
            # check for large asymmetric bulge in upper half of image
            elif pixel_upper == 12 and pixel_lower == 2:
                bulge_array.append(1)
                bulge_locations.append('upper')
            else:
                # if above conditions do not hold, the sequence does not contain a large asymmetric bulge
                bulge_array.append(0)
                bulge_locations.append(0)

    # find the exact location and width of the large asymmetric bulge in the sequence by going over the bulge_array
    widths = []
    bulge_width = 0
    bulge_exact_locations = []
    bulge_exact_location = []
    for i in range(len(bulge_array) - 1):
        # if the integer in the bulge_array is 1, we are at a large asymmetric bulge and we should increment the width
        if bulge_array[i] == 1:
            bulge_width += 1
            bulge_exact_location.append((bulge_locations[i], i))
            # if the next integer in bulge_array is 0, we have reached the end of the bulge and we should store the
            # width and all location info
            if bulge_array[i + 1] == 0:
                widths.append(bulge_width)
                bulge_width = 0
                bulge_exact_locations.append(bulge_exact_location)
                bulge_exact_location = []
            else:
                i += 1

    # create empty values for the attributes of interest if there is no large asymmetric bulge found in the sequence
    if not widths:
        largest_bulge = np.nan
        largest_bulge_location = (np.nan, np.nan)
    # if there is at least one large asymmetric bulge, find the widest one among all and store this as the largest
    # asymmetric bulge of the sequence
    else:
        largest_bulge = np.max(widths)
        largest_bulge_index = np.argmax(widths)
        largest_bulge_location = bulge_exact_locations[largest_bulge_index]
        middle_bulge_location = int(len(largest_bulge_location) / 2)
        largest_bulge_location = (largest_bulge_location[0][0],
                                  largest_bulge_location[middle_bulge_location][1])

    return largest_bulge, largest_bulge_location


def create_annotated_df(dataset, labels):
    """
    :param data_path: path to storage location of dataframe
    :return: dataframe containing paths to all images, the class label associated to the image and information on the
    sequence concepts
    """
    # initialize an empty list that will store all data entries
    tables = []

    for i in tqdm(range(dataset.shape[0])):
        # create a dataframe entry based on the filepath of the image in the dataset folder
        entry = pd.DataFrame([i], columns=['idx'])
        # define the class label and set (train or test) of the entry
        entry['class_label'] = labels[i]
        data = dataset[i]
        # generate all the concept information
        sequence_pairs, stem_begin = pairs_stem(data)
        terminal_loop, loop_start_pixel, loop_highest_row, loop_highest_pixel, loop_length, \
            loop_width, width_gap_start = loop_concepts(data, sequence_pairs)
        ugu_motif_present = ugu_motif(data, terminal_loop, loop_highest_pixel, loop_start_pixel)
        au_pair = AU_pairs_begin_maturemiRNA(sequence_pairs)
        palindrome_score, upper_half_counts, lower_half_counts, len_premiRNA = palindrome(data)
        largest_bulge, largest_bulge_location = large_asymmetric_bulge(data)
        # start adding the concept information to the dataframe entry
        entry['presence_terminal_loop'] = terminal_loop
        entry['start_loop_upperhalf_col'] = loop_start_pixel
        entry['highest_point_loop_upperhalf_row'] = loop_highest_row
        entry['highest_point_loop_upperhalf_col'] = loop_highest_pixel
        entry['loop_length'] = loop_length
        entry['loop_width'] = loop_width
        entry['gap_start'] = width_gap_start
        entry['palindrome_score'] = palindrome_score[1]
        if palindrome_score[0] > 0.6:
            entry['asymmetric'] = True
        else:
            entry['asymmetric'] = False
        entry['large_asymmetric_bulge'] = largest_bulge
        entry['largest_asym_bulge_strand_location'] = largest_bulge_location[0]
        entry['largest_asym_bulge_sequence_location'] = largest_bulge_location[1]
        entry['stem_begin'] = stem_begin
        if np.isnan(loop_start_pixel):
            entry['stem_end'] = 0
            entry['stem_length'] = stem_begin
            entry['total_length'] = stem_begin
            loop_start_pixel = 0
        else:
            entry['stem_end'] = loop_start_pixel + 1
            entry['stem_length'] = stem_begin - loop_width
            entry['total_length'] = stem_begin
        # the stem pairs are defined by the pairs in the pairs array until the start of the terminal loop
        stem_pairs = sequence_pairs[0:len(sequence_pairs) - loop_start_pixel]
        # base pairing propensity: # base pairs (defined by 1 and 2) / stem length
        entry['base_pairs_in_stem'] = (stem_pairs.count(1) + stem_pairs.count(2)) / (stem_begin - loop_start_pixel)
        # base pairing and wobble propensity: # base pairs (defined by 1 and 2) + wobbles (3) / stem length
        entry['base_pairs_wobbles_in_stem'] = (stem_pairs.count(1) + stem_pairs.count(2) + stem_pairs.count(3)) / \
                                              (stem_begin - loop_start_pixel)
        entry['AU_pair_begin_maturemiRNA'] = au_pair
        entry['UGU'] = ugu_motif_present

        # after collecting all concept info for the entry of interest, append it to list containing all data entries
        tables.append(entry)

    # combine all data entries into one dataframe
    dataframe = pd.concat(tables, ignore_index=True)  # create dataframe from list of tables and reset index
    #print(dataframe['class_label'].value_counts())

    return dataframe
