import cv2


def pixel_art(input_file, output_file, pixel_size=16):

    image = cv2.imread(input_file)

    h, w = image.shape[:2]

    small = cv2.resize(

        image,

        (w // pixel_size, h // pixel_size),

        interpolation=cv2.INTER_LINEAR

    )

    result = cv2.resize(

        small,

        (w, h),

        interpolation=cv2.INTER_NEAREST

    )

    cv2.imwrite(output_file, result)