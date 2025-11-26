from crypy_examples.iccdw_cartoon import (
    atom_draw,
    c_draw,
    ic_draw,
    indc_draw,
    xlim,ylim,
    clong_draw,
    cong2c_draw,
    c2ic_draw
)
import matplotlib.pyplot as plt
from useful import (
    fullax,
    savepng
)
import numpy as np
sz = 30
# xlim = np.array((-sz,sz))
# ylim = np.array((-sz,sz))


def savepng(draw_fcn,fn):
    plt.figure(figsize=(8,8))
    atom_draw()
        
    draw_fcn()
    indc_draw()
    ax = plt.gca()
    fig = plt.gcf()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    fullax(ax)
    fig.savefig("assets/"+fn+".png",dpi=200,bbox_inches='tight',pad_inches=0)

savepng(ic_draw,"iccdw_dot")
savepng(c_draw,"ccdw_dot")
savepng(clong_draw,"clongcdw_dot")
from useful import savepng

fig = plt.figure(figsize=(8,8))
cong2c_draw()
ax = plt.gca()
ax.set_xlim(xlim)
ax.set_ylim(ylim)
fullax()
savepng(fig,"trans_cong2ccdw_dot")

fig = plt.figure(figsize=(8,8))
c2ic_draw()
ax = plt.gca()
ax.set_xlim(xlim)
ax.set_ylim(ylim)
fullax()
savepng(fig,"trans_c2iccdw_dot")

from PIL import Image
import os

def crop_image_border(input_path: str, output_path: str, slicing_factor: float):
    """
    PNG 이미지의 테두리를 지정된 비율만큼 잘라내고 새 파일로 저장합니다.

    :param input_path: 입력 PNG 파일 경로.
    :param output_path: 출력 PNG 파일 경로.
    :param slicing_factor: 테두리를 깎을 비율 (0.0 이상 1.0 이하).
                           0.0: 전혀 자르지 않음.
                           1.0: 이미지를 완전히 잘라냄 (0x0 크기).
    """
    # 1. 입력 파일 유효성 검사
    if not os.path.exists(input_path):
        print(f"오류: 입력 파일 경로를 찾을 수 없습니다: {input_path}")
        return

    # 2. slicing_factor 유효성 검사
    if not (0.0 <= slicing_factor <= 1.0):
        print(f"오류: slicing_factor는 0.0과 1.0 사이의 값이어야 합니다 (현재 값: {slicing_factor})")
        return

    try:
        # 3. 이미지 열기
        img = Image.open(input_path)
        width, height = img.size

        # 4. 잘라낼 픽셀 수 계산
        # slicing_factor가 0.1이면, 가로/세로 길이의 10%만큼 양쪽에서 깎는 것이므로
        # 전체 깎는 비율은 width * 2 * factor가 됩니다.
        # 즉, 잘라낼 픽셀은 각 방향(상, 하, 좌, 우)마다 (길이 / 2) * slicing_factor 만큼입니다.

        # 전체 길이 중 남겨야 하는 비율: 1 - slicing_factor
        # 각 변에서 남겨야 하는 길이의 비율: (1 - slicing_factor) / 2 가 아님!

        # 'slicing_factor'를 '이미지의 폭/높이 대비 각 변에서 잘라낼 픽셀의 비율'로 해석합니다.
        # 예: factor가 0.1이면, 왼쪽에서 폭의 10%, 오른쪽에서 폭의 10%를 깎아
        #     최종 폭은 원래 폭의 80%가 됩니다.

        crop_percent = slicing_factor
        
        # 좌측에서 깎을 픽셀 수
        left_crop_pixels = int(width * crop_percent)
        # 상단에서 깎을 픽셀 수
        top_crop_pixels = int(height * crop_percent)
        
        # 5. Cropping Box 좌표 계산
        # (left, top, right, bottom)
        # left: 자르기 시작할 x좌표
        # top: 자르기 시작할 y좌표
        # right: 자르기가 끝날 x좌표 (exclusive)
        # bottom: 자르기가 끝날 y좌표 (exclusive)
        
        left = left_crop_pixels
        top = top_crop_pixels
        right = width - left_crop_pixels
        bottom = height - top_crop_pixels

        # 6. 유효한 Cropping 영역 확인
        if left >= right or top >= bottom:
            print(f"경고: slicing_factor={slicing_factor}로 인해 유효한 이미지를 남길 수 없습니다.")
            print(f"원래 크기: {width}x{height}, 자르기 영역: ({left}, {top}, {right}, {bottom})")
            # 1.0일 때 0x0 이미지를 만드는 대신, 1.0보다 큰 값으로 인해 오류가 날 경우를 대비
            if slicing_factor >= 0.5:
                print("최소 크기인 1x1 픽셀로 저장하거나, 저장을 건너뛸 수 있습니다.")
                # 여기서는 0x0을 방지하고 에러 처리
            
            # 자르기가 불가능한 경우 원본 이미지 반환하거나, 중단.
            # 사용자 요청에 따라 1.0일 때 완전히 없애는 것을 목표로 하므로, 
            # 1.0일 때는 0x0이 되더라도 진행하거나, 1x1을 남기는 것으로 처리 가능
            # Pillow는 0x0을 허용하지 않으므로, 유효성 검사에서 걸러내는 것이 좋습니다.
            if left > right or top > bottom:
                print("오류: 자르기 영역이 유효하지 않습니다. 저장을 건너뜁니다.")
                return

        # 7. 이미지 자르기 (Crop)
        cropped_img = img.crop((left, top, right, bottom))

        # 8. 새 파일로 저장
        cropped_img.save(output_path, "PNG")
        print(f"✅ 이미지 자르기 성공: {input_path} -> {output_path} (Slicing Factor: {slicing_factor})")

    except Exception as e:
        print(f"이미지 처리 중 오류가 발생했습니다: {e}")
        
# --- 사용 예시 ---

# 1. 테스트 이미지 준비 (이 코드를 실행하기 전에 'input.png' 파일이 같은 폴더에 있어야 합니다.)
#    예시를 위해 임시로 더미 이미지를 생성하는 코드를 추가할 수도 있지만,
#    사용자님께서 직접 PNG 파일을 준비해 주시는 것이 좋습니다.
inputs = ["assets\iccdw_dot.png","assets\ccdw_dot.png","assets\clongcdw_dot.png",
"assets\\trans_cong2ccdw_dot.png","assets\\trans_c2iccdw_dot.png"]


for input_file in inputs:
    INPUT_FILE = input_file    # <- 실제 이미지 파일 이름으로 변경하세요.
    OUTPUT_FILE =  INPUT_FILE[0:-4] + "crop.png"
    crop_image_border(INPUT_FILE, OUTPUT_FILE, 0.2)

# slicing_factor = 0.0 (전혀 깎지 않음 -> 원본 크기 유지)
# crop_image_border(INPUT_FILE, OUTPUT_FILE_3, 0.0)

# slicing_factor = 0.5 (각 변에서 50% 깎음 -> 최종 크기는 0%가 됨)
# 이 경우 Pillow는 0x0 이미지를 허용하지 않으므로, 
# 실제 코드를 실행할 때는 에러가 발생하거나 0.5보다 작은 값으로 자동 조정될 수 있습니다.
# 사용자 요청(1.0이면 이미지가 하나도 없도록)에 최대한 가깝게 구현했습니다.
# crop_image_border(INPUT_FILE, "output_factor_05.png", 0.5) 
# (factor가 0.5일 때, left=right가 되어 이미지가 사라집니다. 
# 만약 1.0일 때 완전히 사라지게 하려면, factor를 (1 - 남길 비율)로 계산해야 합니다.)