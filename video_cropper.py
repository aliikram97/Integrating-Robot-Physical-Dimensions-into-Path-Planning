import cv2


def crop_video(input_file, output_file, start_time, end_time, fps):
    video_capture = cv2.VideoCapture(input_file)
    fps_input = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_time * fps_input)
    end_frame = int(end_time * fps_input)

    video_writer = cv2.VideoWriter(output_file,
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps,
                                   (width, height))

    frame_count = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()

        if not ret:
            break

        if frame_count >= start_frame and frame_count <= end_frame:
            video_writer.write(frame)

        if frame_count > end_frame:
            break

        frame_count += 1

    video_capture.release()
    video_writer.release()


# Example usage:
input_file = r'C:\Users\HomePC\Downloads\experiment-20231213T094822Z-001\experiment\exp2/live3.mp4'  # Replace with your input video file
output_file = r'C:\Users\HomePC\Downloads\experiment-20231213T094822Z-001\experiment\exp2/sucess_experiment.mp4'  # Replace with the desired output file name
start_time = 343  # Start time in seconds
end_time = 429  # End time in seconds
fps = 20  # Desired frames per second

crop_video(input_file, output_file, start_time, end_time, fps)
