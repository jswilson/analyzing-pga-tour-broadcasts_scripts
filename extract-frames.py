import base64
import json
import os
import sys

from openshot import FFmpegReader
from PIL import Image

class VideoFile:
    def __init__(self, filename):
        self.filename = filename

        try:
            self.capture = FFmpegReader(self.filename)
        except:
            raise CouldNotOpenVideoException

        self.capture.Open()

    def get_frame_as_image(self, frame_number):
        """Returns a PIL image at the given frame number"""
        frame = self.capture.GetFrame(frame_number+1) # openshot is 1-indexed

        data = base64.b64decode(frame.GetImageAsString())
        frame.DeleteImageAsStringMemory()
        image = Image.frombytes('RGBA', (frame.GetWidth(), frame.GetHeight()), data)
        image = image.convert('RGB')
        return image

    def get_total_frames(self):
        """Returns the total number of frames contained in this video"""
        return int(json.loads(self.capture.Json())['video_length'])

class CouldNotOpenVideoException(Exception):
    pass

def _create_destination_directory(dir):
    """Creates the destination directory where we are writing the frames"""
    if not os.path.exists(dest):
        try:
            os.makedirs(dest)
        except:
            print('Could not create destination directory!')
            sys.exit(0)

def _write_frames(start_frame, increment, video_file, destination, file_prefix=''):
    """Writes selected video frames to image files on disk"""
    curr = start_frame
    while curr < video_file.get_total_frames():
        image = video_file.get_frame_as_image(curr)

        size = 224, 224
        image.thumbnail(size, Image.ANTIALIAS)

        background = Image.new('RGB', size, (255, 255, 255))
        background.paste(
            image, (int((size[0] - image.size[0]) / 2), int((size[1] - image.size[1]) / 2))
        )

        background.save(os.path.join(destination, file_prefix + str(curr)) + '.jpg')

        curr += increment
        if curr % 30000 == 0:
            print(curr/30)


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print('Must provide two arguments: a path to a video file, and a directory to write frames!')
        sys.exit(0)

    starting_frame = 0
    increment = 30  #30 gives us about 1 frame per second
    video_file = VideoFile(sys.argv[1])
    dest = sys.argv[2]

    _create_destination_directory(dest)
    _write_frames(starting_frame, increment, video_file, dest, file_prefix=sys.argv[3])
