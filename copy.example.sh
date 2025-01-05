#!/bin/bash

# Set username and password as variables
username=""
password=""

# Set the destination folder on the remote server
destination_folder=""

# Clear the contents of the destination folder on the remote server
sshpass -p "$password" ssh "$username"@s7edu.di.uminho.pt "rm -rf ~/$destination_folder/*"

# Copy the contents of the current folder to the remote server
rsync -avz -e "sshpass -p '$password' ssh" --exclude '.venv/' ./ "$username"@s7edu.di.uminho.pt:~/$destination_folder/

# SSH into the remote server and execute the desired commands
sshpass -p "$password" ssh "$username"@s7edu.di.uminho.pt << EOF
cd ~/$destination_folder/
make clean
EOF