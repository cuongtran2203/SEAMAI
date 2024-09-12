#!/bin/sh

USER=simexp
HOST=step2bmt05.xdata.nict.jp
KEY=privatekey/simexp5.txt

echo "connecting: ${HOST}"
ssh ${USER}@${HOST} -L8888:localhost:8888 -i ${KEY}
