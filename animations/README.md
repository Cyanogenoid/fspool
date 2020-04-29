# Animations for ICLR 2020 video

This directory contains the manim script to create the animations used in my ICLR 2020 video.
I hope that this will be a useful resource for people who want to make animations for their own paper.

## Basic instructions
- Install manim https://github.com/3b1b/manim
- `python -m manim test.py --high_quality`

## Tips for video production
To turn the rendered scenes into a video, I recommend using an external video editor.
I used DaVinci Resolve, which is free but probably a bit overkill for the task.

The resulting scenes will have no breaks for talking between them.
To sync them up with the talking pace, I recommend making the following change to manim to extend every animation by about 2 frames so that they are easier to cut in the video editing.

```diff
diff --git a/manimlib/scene/scene.py b/manimlib/scene/scene.py
index 014ea528..70ea675d 100644
--- a/manimlib/scene/scene.py
+++ b/manimlib/scene/scene.py
@@ -310,7 +310,7 @@ class Scene(Container):
         return time_progression

     def get_run_time(self, animations):
-        return np.max([animation.run_time for animation in animations])
+        return np.max([animation.run_time for animation in animations]) + 0.1
```

Then, import all the partial video files in the `media/videos/test/1080p60/partial_movie_files` directory into your video editor and freeze frame the last frame of each animation.
Now you can extend the duration of the freeze frames to match the talking pace.
