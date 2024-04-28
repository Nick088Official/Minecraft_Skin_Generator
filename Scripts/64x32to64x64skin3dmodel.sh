convert "$1" -background transparent -extent 64x64 \
	\( "$1" -crop 4x4+4+16 -flop \) -geometry +20+48 -composite \
	\( "$1" -crop 4x4+8+16 -flop \) -geometry +24+48 -composite \
	\( "$1" -crop 4x12+8+20 -flop \) -geometry +16+52 -composite \
	\( "$1" -crop 4x12+4+20 -flop \) -geometry +20+52 -composite \
	\( "$1" -crop 4x12+0+20 -flop \) -geometry +24+52 -composite \
	\( "$1" -crop 4x12+12+20 -flop \) -geometry +28+52 -composite \
	\( "$1" -crop 4x4+44+16 -flop \) -geometry +36+48 -composite \
	\( "$1" -crop 4x4+48+16 -flop \) -geometry +40+48 -composite \
	\( "$1" -crop 4x12+48+20 -flop \) -geometry +32+52 -composite \
	\( "$1" -crop 4x12+44+20 -flop \) -geometry +36+52 -composite \
	\( "$1" -crop 4x12+40+20 -flop \) -geometry +40+52 -composite \
	\( "$1" -crop 4x12+52+20 -flop \) -geometry +44+52 -composite \
"$1-converted.png"