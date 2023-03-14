import logging
from multiprocessing import Process, Queue
from pathlib import Path

import boto3
import cv2
import numpy as np
import qrcode

from squat_wars.game_state import WINDOW_HEIGHT, WINDOW_WIDTH
from squat_wars.player import Player


class VideoWriter(Process):
    def __init__(self, input_queue: Queue, out_queue: Queue):
        self.input_queue = input_queue
        self.out_queue = out_queue
        super().__init__()

    def upload_video_file(self, file_name: str, bucket_name: str, object_name: str) -> bool:
        """Upload a video file to an S3 bucket

        :param file_name: File to upload
        :param bucket_name: Bucket to upload to
        :param object_name: S3 object name
        :return: True if file was uploaded, else False
        """

        # Upload the file
        try:
            self.s3_client.upload_file(
                file_name, bucket_name, object_name, ExtraArgs={"ContentType": "video/mp4"}
            )
        except Exception as e:
            logging.error(e)
            return False
        return True

    def create_presigned_url(self, bucket_name: str, object_name: str, expiration=60 * 60 * 24 * 7):
        """Generate a presigned URL to share an S3 object

        :param bucket_name: Bucket where the object is in
        :param object_name: S3 object name
        :param expiration: Time in seconds for the presigned URL to remain valid
        :return: Presigned URL as string. If error, returns None.
        """

        # Generate a presigned URL for the S3 object
        try:
            response = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket_name, "Key": object_name},
                ExpiresIn=expiration,
            )
        except Exception as e:
            logging.error(e)
            return None

        # The response contains the presigned URL
        return response

    def run(self):
        """
        Loop that waits indefinitely for frames until a Player is received. When that occurs
        a video is created and uploaded to a S3 bucket. Finally, a QR code is created
        containing the link to the uploaded video.
        """

        self.s3_client = boto3.client("s3")
        s3_bucket = "tryolabs-khipu-2023"
        frames = []

        while True:
            # load input frames and calculate video fps
            res = self.input_queue.get()

            if isinstance(res, Player):
                fps = int((len(frames) - 1) / 10)

                # add final screen to video for 5 sec
                frames.extend([frames[-1]] * (5 * fps - 1))

                # create folder for videos
                folder = Path("videos")
                folder.mkdir(exist_ok=True)

                # create video
                filename = f"{res.timestamp}_{res.email}.mp4"
                video_file = folder / filename
                fourcc = cv2.VideoWriter_fourcc("a", "v", "c", "1")
                writer = cv2.VideoWriter(
                    str(video_file), fourcc, fps, (WINDOW_WIDTH, WINDOW_HEIGHT)
                )
                for frame in frames:
                    writer.write(frame)
                writer.release()

                # upload video to s3
                s3_filename = filename
                is_uploaded = self.upload_video_file(str(video_file), s3_bucket, s3_filename)
                if is_uploaded:
                    video_file.unlink()

                # get url to s3 object
                url = self.create_presigned_url(s3_bucket, s3_filename)
                if url is None:
                    url = "tryolabs.com/404"

                # create qrcode
                qr = qrcode.make(data=url, version=1, box_size=5, border=1).convert("RGB")
                qr.save("qr_code.png")
                self.out_queue.put(np.array(qr))

                # reset frames
                frames = []
            else:
                frames.append(res)
