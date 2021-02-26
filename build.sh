#!/bin/bash

download_link=https://github.com/ArjunSahlot/find_object/archive/main.zip
temporary_dir=$(mktemp -d) \
&& curl -LO $download_link \
&& unzip -d $temporary_dir main.zip \
&& rm -rf main.zip \
&& mv $temporary_dir/find_object-main $1/find_object \
&& rm -rf $temporary_dir
echo -e "[0;32mSuccessfully downloaded to $1/find_object[0m"
