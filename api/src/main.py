import cv2
import numpy as np
from fastapi import FastAPI, Response
from fastapi import Request, File
import paramiko
from fastapi.responses import StreamingResponse

import time
from detection_service import DetectionEngine
from recognition_service import RecognitionEngine
from util import prepare_for_detector, nms_np, preprocess_image_recognizer
from template_matching import TemplateMatching
from constants import DETECTION_IMAGE_H, DETECTION_IMAGE_W
from datetime import datetime


app = FastAPI()
detector = DetectionEngine()
recognizer = RecognitionEngine()
template_matcher = TemplateMatching()

def getPlates(image, img_orig, img_model):
    plate_output = detector.predict(img_model)
    plates = nms_np(plate_output[0], conf_thres=0.7, include_conf=True)
    height_orig, width_orig, _ = image.shape
    ratio_width = width_orig/DETECTION_IMAGE_W
    ratio_height = height_orig/DETECTION_IMAGE_H
    if len(plates) > 0:
        plates[..., [4, 6, 8, 10]] += plates[..., [0]]
        plates[..., [5, 7, 9, 11]] += plates[..., [1]]
        ind = np.argsort(plates[..., -1])
        plates = plates[ind]
        plate = plates[0]
        box = np.copy(plate[:12]).reshape(6, 2)
        for i in range(6):
            box[i, 0] = box[i, 0]*ratio_width
            box[i, 1] = box[i, 1]*ratio_height
        plate_img, is_squared = preprocess_image_recognizer(image, box)
        plate_labels, probs, country_code = recognizer.predict(plate_img)
        if is_squared:
            plate_label, country_code = template_matcher.process_square_lp(plate_labels[0], plate_labels[1])
            prob = probs[0]*probs[1]
            return [plate_label], [prob], country_code
        else:
            return plate_labels, probs, country_code
    else:
        return None, None, None


def readb64(uri):
    nparr = np.fromstring(uri, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


@app.post("/")
async def main():
    return {"success": True}


@app.post("/api/image_test")
async def analyze_route(request: Request):
    form = await request.form()
    try:
        t1 = time.time()
        images = []
        results = []
        label_bool = False
        # upload_file = form["image"]
        # filename = form["image"].filename  # str
        for i in range(5):
            try:
                name = f"image_{i}"
                # name = "image_0"
                images = await form[name].read()
            except Exception:
                break
        # for i in range(len(images)):
            try:
                image_base64 = images # bytes
                # content_type = form["image"].content_type  # str
                image = readb64(image_base64)
                img_orig, img_model = prepare_for_detector(image)

                label_bool = None
            except:
                results.append({"status": False, "message": "incorrect image"})
                continue
            label, probs, country_code = getPlates(image, img_orig, img_model)
            t2 = time.time()
            process_time = t2 - t1
            if label is None:
                results.append({"status": False, "message": "no plate in image"})
            else:
                # label_bool = True
                results.append({"status": True, "label": label, "prob": probs, "country_code": country_code,
                                "exec_time": process_time})
        # if label_bool is False:
        #     print(f"status: {False}, message: no plate in image")
        #     return {"status": False, "message": "no plate in image"}
        # else:
        #     print(f"status: {True}, label: {label}, prob: {probs}, country_code: {country_code},"
        #           f"exec_time: {process_time}")
        print(results)
        return results
    except Exception as e:
        print(f"state: {False}, message: unknown error")
        return {"status": False, "message": "unknown error", "body": request.body, "head": request.headers}

@app.post("/api/image")
async def analyze_route(request: Request):
    form = await request.form()
    try:
        if "image" in form:
            t1 = time.time()
            # upload_file = form["image"]
            # filename = form["image"].filename  # str
            try:
                image_base64 = await form["image"].read()  # bytes
                # content_type = form["image"].content_type  # str
                image = readb64(image_base64)
                img_orig, img_model = prepare_for_detector(image)
            except:
                return {"status": False, "message": "incorrect image"}
            label, probs, country_code = getPlates(image, img_orig, img_model)
            t2 = time.time()
            process_time = t2-t1
            if label is None and probs is None:
                return {"status": False, "message": "no plate in image"}
            else:
                print(f"status: {True}, label: {label}, prob: {probs}, country_code: {country_code},"
                      f"exec_time: {process_time}")
                return {"status": True, "label": label, "prob": probs, "country_code": country_code,
                        "exec_time": process_time}
        else:
            return {"status": False, "message": "no image in request body",
                    "body": request.body, "head": request.headers, "form": request.form}
    except Exception as e:
        return {"status": False, "message": "unknown error", "body": request.body, "head": request.headers}

@app.get("/api/logs_local")
async def logs_local(start: str = '2022-09-27T16:00:00', finish: str = '2022-09-27T16:30:00',
                        camera_ip: str = '172.27.14.180', ip_local: str = '10.66.16.4'):
    # form = await request.form()

    # ip_local = "10.66.16.4"
    username = "nsagitzhan"


    jhost = paramiko.SSHClient()
    jhost.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    jhost.connect(ip_local, username=username, key_filename="/home/nsagitzhan/.ssh/id_ed25519")

    stdin, stdout, stderr = jhost.exec_command(f'docker logs -t recognizer --since {start} --until {finish}  | '
                                               f'sed -n -e "/Car Tracker {camera_ip}: tracking new car /, /Package/p" '
                                               f'| grep "Car : plate: " | sed "s/Z Car/Z, Car/g" ')  # edited#

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%YT%H:%M:%S")
    a = stdout.readlines()#.decode("utf-8")
    response = StreamingResponse(iter(a), media_type="text/csv")
    response.headers["Content-Disposition"] = f"attachment; filename=object_recognizer_{dt_string}.csv"
    jhost.close()

    return response

@app.get("/api/logs_cloud")
async def logs_cloud(start: str = '2022-09-27T16:00:00', finish: str = '2022-09-27T16:30:00'):
    # form = await request.form()

    ip_cloud = "10.66.100.185"
    username = "nsagitzhan"
    password_cloud = "1995"

    vm = paramiko.SSHClient()
    vm.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    vm.connect(ip_cloud, username=username, password=password_cloud)

    stdin, stdout, stderr = vm.exec_command(f'docker logs -t recognizer_api --since {start} --until {finish} 2>&1 '
                                            f'| grep "status: True" ')  # edited#

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%YT%H:%M:%S")
    a = stdout.readlines()#.decode("utf-8")
    print(a)
    response = StreamingResponse(iter(a), media_type="text/csv")
    response.headers["Content-Disposition"] = f"attachment; filename=cloud_recognizer_{dt_string}.csv"
    vm.close()

    return response

@app.get("/api/stats")
async def analyze_route(start: str = '2022-09-27T16:00:00', finish: str = '2022-09-27T16:30:00'):
    # form = await request.form()

    ip_cloud = "10.66.100.185"
    username = "nsagitzhan"
    password_cloud = "1995"

    vm = paramiko.SSHClient()
    vm.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    vm.connect(ip_cloud, username=username, password=password_cloud)

    stdin, stdout, stderr = vm.exec_command(f'docker logs -t recognizer_api --since {start} --until {finish} 2>&1 '
                                            f'| grep "status: True" | wc -l')  # edited#

    count_success = stdout.read()#.decode("utf-8")

    stdin, stdout, stderr = vm.exec_command(f'docker logs -t recognizer_api --since {start} --until {finish} 2>&1 '
                                            f'| grep "incorrect image" | wc -l')  # edited#

    count_incorrect = stdout.read()  # .decode("utf-8")

    stdin, stdout, stderr = vm.exec_command(f'docker logs -t recognizer_api --since {start} --until {finish} 2>&1 '
                                            f'| grep "no plate in image" | wc -l')  # edited#

    count_no_plate = stdout.read()  # .decode("utf-8")

    stdin, stdout, stderr = vm.exec_command(f'docker logs -t recognizer_api --since {start} --until {finish} 2>&1 '
                                            f'| grep "no image in request body" | wc -l')  # edited#

    count_no_image = stdout.read()  # .decode("utf-8")

    stdin, stdout, stderr = vm.exec_command(f'docker logs -t recognizer_api --since {start} --until {finish} 2>&1 '
                                            f'| grep "unknown error" | wc -l')  # edited#

    count_unknown = stdout.read()  # .decode("utf-8")

    count_total = int(count_success)+int(count_incorrect)+int(count_unknown)+int(count_no_image)+int(count_no_plate)

    response = f"total requests: {count_total}", \
               f"total success: {int(count_success)}," \
               f"improper images: {int(count_incorrect)}," \
               f"images without plate: {int(count_no_plate)}," \
               f"requests without image: {int(count_no_image)}," \
               f"unknown errors: {int(count_unknown)}"
    vm.close()

    return response
