# Image-Matching

Program to match face of a person with images present in Database.

---

### Dependencies

The list of required libraries can be found in the requirements.txt file.
You can use pip to install the packages from requirements.txt file.

Note: The code is working for tensorflow 1.15 but to work on tensorflow 2.0 you will have to make some changes to the code.

```
pip3 install -r requirements.txt
```

### Installation

1. Clone the git repo

```
git clone https://github.com/data-stats/Roshan-Mishra.git
```

2. Change directory

```
cd Roshan-Mishra/Image-Matching
```

3. You will have to first run the single_image.py file to save a single image of yours to the database.

```
python3 single-image.py
```

4. Now run the image matching file which uses your webcam and detects faces to match them with the ones already present in database.

```
python3 image-matching.py
```
