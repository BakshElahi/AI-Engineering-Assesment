# Write-up



## 1. Timings 
My initial version was only tested on 4 videos, not 50 files. Using FP32 it took 909 seconds. The time was reduced to 398 seconds after improvements with FP16 and 142 seconds for every third frame. Based on these tests, I estimated that the original version would take around 28 hours for 50 files. The final version was made with two T4 GPUs and all 50 files. The total time was approximately 5 hours and 42 minutes, with notebook A taking 10,357.1 seconds and notebook B taking 10,143 seconds. The fastest time was obtained by skipping frames.

## 2. Pipeline 
The script starts by reading each Parquet file and gets the video data from the column named `mp4`. It then converts the video into frames, and calculates the seconds for each frame based on the time of the first frame. To reduce processing time it chooses every third frame and sends the selected frames in chunks of 32 to YOLO. YOLO is able to identify five objects: persons, vehicles, dogs, laptop computers, and mobile phones. The script is able to output the number of objects detected and the median of people for every second. If the object is detected in at least half of the sampled frames during that second, it is counted. Errors that are detected or occur during decoding of a video are saved to the error file without aborting the overall process. Lastly, the script dumps the individual output files together in a single file and renders the results in JSONL format.

## 3. Where the time goes
In my final version, most of the processing time is not decodng video, but inferring using YOLO. I did this with the timing info I had in my run.py script and counters from YOLO. It takes around 7.4ms to make an inference and 0.56ms for preprocessing and postprocessing for each frame according to `r.speed`. This indicates that the inference step is the major time-consuming part of this. I also recorded the wait_decode counter which was used to measure the wait time for decoded frames. The total processing time was around 800 seconds, and the processing time around 1-3 seconds per file. Also, I did my own decoding test and got about 1170 fps with one thread. From these results, it can be concluded that decoding is not a major constraint. The majority of the time is used for running the object detection model on the selected frames.

## 4. Checking correctness 
To be sure that the optimizations didn't cause any loss of accuracy, I used my original version that performed on FP32 with all frames as a reference and compared the results using a comparison function. I looked at three measures: the detection F1 score, the actual number of people detected, and the number of people detected within one of the actual numbers. The FP16 version yielded nearly the same results and stride 3 was also high enough on all thresholds. Then I reviewed the final output of 50 files, 1650 videos, for any missing or error videos. Lastly, the script was run again with locked requirements with no differences greater than 0.2% found. But my reference was a self-generated one.

## 5. Working with AI
An AI recommendation was to use the batch size of 8 for optimum performance. But, looking at the benchmark results, I saw that the processing time for the different batch sizes was very similar and that there was no difference in memory used. This appeared to be odd, and by further investigation I realized that the `batch` argument was not being used when frames were passed as a list. All of the tests were, in effect, executed in the same manner. I declined the suggestion due to the fact that the benchmark was not valid. This taught me about testing out AI suggestions, which is crucial.

## 6. Next steps
If I had an extra week, I would test out the stride setting with more videos to ensure I'm not losing what is important and also ensure people count doesn't change. Also, I would try the TensorRT version on a larger set of videos to determine if the accelerated performance is consistent. As of yet, there have been good results, but with more data tested, I would be more confident in the pipeline's speed and its accuracy. This would make sure that, as well as on the small test sample, this optimized version works well on other videos.

