from flask import Flask, jsonify, request
from PIL import Image
import numpy as np
import io
from io import BytesIO
import cv2
from flask import request, make_response
from pyngrok import ngrok
import requests
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import requests
from io import BytesIO
import base64
import sys
import math
from PIL import Image
import requests
from io import StringIO
from flask import request, make_response
THRESHOLD_RATIO = 2000
MIN_AVG_RED = 60
MAX_HUE_SHIFT = 120
BLUE_MAGIC_VALUE = 1.2

app = Flask(__name__)

# This function takes an image matrix and hue shift angle in degrees as input
def hue_shift_red(mat, h):

    # Calculate the sine and cosine values of the hue shift angle in radians
    U = math.cos(h * math.pi / 180)
    W = math.sin(h * math.pi / 180)

    # Apply color transformation using coefficients derived from the hue shift angle
    r = (0.299 + 0.701 * U + 0.168 * W) * mat[..., 0]
    g = (0.587 - 0.587 * U + 0.330 * W) * mat[..., 1]
    b = (0.114 - 0.114 * U - 0.497 * W) * mat[..., 2]

    # Stack the color channels back together to form the filtered image
    return np.dstack([r, g, b])

# This function takes an array of numerical values as input
def normalizing_interval(array):

    # Set initial values for high, low, and max distance between values in the array
    high = 255
    low = 0
    max_dist = 0

    # Loop through the array and find the largest distance between adjacent values
    for i in range(1, len(array)):
        dist = array[i] - array[i-1]
        if(dist > max_dist):
            max_dist = dist
            high = array[i]
            low = array[i-1]

    # Return the high and low values that represent the normalizing interval for the array
    return (low, high)

# This function takes an image matrix and a filter matrix as input
def apply_filter(mat, filt):

    # Separate the color channels from the image matrix
    r = mat[..., 0]
    g = mat[..., 1]
    b = mat[..., 2]

    # Apply the filter matrix to each color channel and add an offset to each channel
    r = r * filt[0] + g*filt[1] + b*filt[2] + filt[4]*255
    g = g * filt[6] + filt[9] * 255
    b = b * filt[12] + filt[14] * 255

    # Stack the color channels back together and clip any values outside the range [0, 255]
    filtered_mat = np.dstack([r, g, b])
    filtered_mat = np.clip(filtered_mat, 0, 255).astype(np.uint8)

    # Return the filtered image matrix
    return filtered_mat


def get_filter_matrix(mat):

    mat = cv2.resize(mat, (256, 256))

    # Get average values of RGB
    avg_mat = np.array(cv2.mean(mat)[:3], dtype=np.uint8)
    
    # Find hue shift so that average red reaches MIN_AVG_RED
    new_avg_r = avg_mat[0]
    hue_shift = 0
    while(new_avg_r < MIN_AVG_RED):

        shifted = hue_shift_red(avg_mat, hue_shift)
        new_avg_r = np.sum(shifted)
        hue_shift += 1
        if hue_shift > MAX_HUE_SHIFT:
            new_avg_r = MIN_AVG_RED

    # Apply hue shift to whole image and replace red channel
    shifted_mat = hue_shift_red(mat, hue_shift)
    new_r_channel = np.sum(shifted_mat, axis=2)
    new_r_channel = np.clip(new_r_channel, 0, 255)
    mat[..., 0] = new_r_channel

    # Get histogram of all channels
    hist_r = hist = cv2.calcHist([mat], [0], None, [256], [0,256])
    hist_g = hist = cv2.calcHist([mat], [1], None, [256], [0,256])
    hist_b = hist = cv2.calcHist([mat], [2], None, [256], [0,256])

    normalize_mat = np.zeros((256, 3))
    threshold_level = (mat.shape[0]*mat.shape[1])/THRESHOLD_RATIO
    for x in range(256):
        
        if hist_r[x] < threshold_level:
            normalize_mat[x][0] = x

        if hist_g[x] < threshold_level:
            normalize_mat[x][1] = x

        if hist_b[x] < threshold_level:
            normalize_mat[x][2] = x

    normalize_mat[255][0] = 255
    normalize_mat[255][1] = 255
    normalize_mat[255][2] = 255

    adjust_r_low, adjust_r_high = normalizing_interval(normalize_mat[..., 0])
    adjust_g_low, adjust_g_high = normalizing_interval(normalize_mat[..., 1])
    adjust_b_low, adjust_b_high = normalizing_interval(normalize_mat[..., 2])


    shifted = hue_shift_red(np.array([1, 1, 1]), hue_shift)
    shifted_r, shifted_g, shifted_b = shifted[0][0]

    red_gain = 256 / (adjust_r_high - adjust_r_low)
    green_gain = 256 / (adjust_g_high - adjust_g_low)
    blue_gain = 256 / (adjust_b_high - adjust_b_low)

    redOffset = (-adjust_r_low / 256) * red_gain
    greenOffset = (-adjust_g_low / 256) * green_gain
    blueOffset = (-adjust_b_low / 256) * blue_gain

    adjust_red = shifted_r * red_gain
    adjust_red_green = shifted_g * red_gain
    adjust_red_blue = shifted_b * red_gain * BLUE_MAGIC_VALUE

    return np.array([
        adjust_red, adjust_red_green, adjust_red_blue, 0, redOffset,
        0, green_gain, 0, 0, greenOffset,
        0, 0, blue_gain, 0, blueOffset,
        0, 0, 0, 1, 0,
    ])

def correct(mat):
    original_mat = mat.copy()

    filter_matrix = get_filter_matrix(mat)
    
    corrected_mat = apply_filter(original_mat, filter_matrix)
    corrected_mat = cv2.cvtColor(corrected_mat, cv2.COLOR_RGB2BGR)

    return corrected_mat


def singleScaleRetinex(img, sigma):
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))
    return retinex

def multiScaleRetinex(img, sigma_list):
    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        retinex += singleScaleRetinex(img, sigma)
    retinex = retinex / len(sigma_list)
    return retinex

def colorRestoration(img, alpha, beta):
    img_sum = np.sum(img, axis=2, keepdims=True)
    color_restoration = beta * (np.log10(alpha * img) - np.log10(img_sum))
    return color_restoration

def simplestColorBalance(img, low_clip, high_clip):    
    total = img.shape[0] * img.shape[1]
    for i in range(img.shape[2]):
        unique, counts = np.unique(img[:, :, i], return_counts=True)
        current = 0
        for u, c in zip(unique, counts):            
            if float(current) / total < low_clip:
                low_val = u
            if float(current) / total < high_clip:
                high_val = u
            current += c
                
        img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)
    return img  

def MSRCR(img, sigma_list, G, b, alpha, beta, low_clip, high_clip):
    img = np.float64(img) + 1.0
    img_retinex = multiScaleRetinex(img, sigma_list)    
    img_color = colorRestoration(img, alpha, beta)    
    img_msrcr = G * (img_retinex * img_color + b)
    for i in range(img_msrcr.shape[2]):
        img_msrcr[:, :, i] = (img_msrcr[:, :, i] - np.min(img_msrcr[:, :, i])) / \
                             (np.max(img_msrcr[:, :, i]) - np.min(img_msrcr[:, :, i])) * \
                             255
    img_msrcr = np.uint8(np.minimum(np.maximum(img_msrcr, 0), 255))
    img_msrcr = simplestColorBalance(img_msrcr, low_clip, high_clip)       
    return img_msrcr

@app.route('/dehaze', methods=['POST'])
def msrcr():
    image_b64 = request.json['image_b64']
    image_data = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(image_data)).convert('RGB')
    sigma_list = [15, 80, 250]
    G = 5
    b = 25
    alpha = 125
    beta = 46
    low_clip = 0.01
    high_clip = 0.99
    img = np.array(img)
    img_msrcr = MSRCR(img, sigma_list, G, b, alpha, beta, low_clip, high_clip)
    buffer = BytesIO()
    img_msrcr = Image.fromarray(img_msrcr)
    img_msrcr.save(buffer, format='JPEG')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    response = {'image': img_str}
    return response

@app.route('/color-correction', methods=['POST'])
def correct_image():
    image_url = request.json['image_url']
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    mat = np.array(img)
    rgb_mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
    
    corrected_mat = correct(rgb_mat)
    buffer = BytesIO()
    img_fin = Image.fromarray(corrected_mat)
    img_fin.save(buffer, format='JPEG')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    response = {'image': img_str}
    return response

@app.route('/denoise', methods=['POST'])
def denoise_image():
    # Get the base64 encoded image from the request
    image_base64 = request.json['image_base64']
    
    # Convert the base64 encoded image to numpy array
    image_bytes = base64.b64decode(image_base64)
    image_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
    
    # Denoise the image using the fastNlMeansDenoisingColored function from OpenCV
    denoised_img = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 3, 10)
    
    # Convert the denoised image to JPEG format and encode it in base64
    _, jpeg_img = cv2.imencode('.jpg', denoised_img)
    denoised_img_base64 = base64.b64encode(jpeg_img).decode('utf-8')
    
    # Return the denoised image as a response
    return jsonify({'image': denoised_img_base64})


if __name__ == '__main__':
    public_url = ngrok.connect(port=5000)
    print(' * Tunnel URL:', public_url)
    app.run(port=80)
