#!/usr/bin/env bash

set -eu -o pipefail

if [ -d left ] || [ -d right ]; then
  echo "the directories 'left' and 'right' must not exist! please remove them first!" >2
  exit 1
fi

mkdir left right

mv cam00*.png left/
mv cam01*.png right/

for folder in left right ; do
  pushd folder
  for fl in cam*.png ; do
    new_fl=$(echo $fl | sed -r 's/cam0._image([0-9]+).png/0\1.png/g')
    mv "$fl" "$new_fl"
  done
  popd
done

exit 0
