CURR_DIR=`pwd`

NOTE_PATH="$(pwd)/_ipynb"
POST_PATH="$(pwd)/_posts"
IMG_PATH="$(pwd)/assets/img"

FILE_NAME="$1"
FILE_BASE=`basename $FILE_NAME .ipynb`

POST_NAME="${FILE_BASE}.md"
IMG_NAME="${FILE_BASE}_files"

POST_DATE_NAME=`date "+%Y-%m-%d-"`${POST_NAME}

cd $NOTE_PATH
# convert the notebook
jupyter nbconvert --to markdown $FILE_NAME

# change image paths
#sed -i "s/[png](/[png](/g" $POST_NAME

# move everything to blog area
mv $POST_NAME "${POST_PATH}/${POST_DATE_NAME}"
if [ -d "${IMG_PATH}/${IMG_NAME}" ]; then rm -Rf "${IMG_PATH}/${IMG_NAME}"; fi
mv $IMG_NAME "${IMG_PATH}/"

cd $CURR_DIR
python "change_img_path.py" "${POST_PATH}/${POST_DATE_NAME}"

# add files to git repo to be included in next commit
cd $POST_PATH
git add $POST_DATE_NAME
cd $IMG_PATH
git add $IMG_NAME

# make git submission
cd $CURR_DIR
git commit -m "\"New blog entry ${FILE_BASE}\""

# push changes to server
git push