from PIL import Image
import os
from typing import List

def create_animated_gif(
    input_png_paths: List[str], 
    output_gif_path: str, 
    duration_ms: int = 100,
    loop: int = 0
):
    """
    여러 PNG 파일을 입력받아 Animated GIF로 변환하여 저장합니다.

    :param input_png_paths: 입력 PNG 파일들의 경로 목록 (프레임 순서대로).
    :param output_gif_path: 출력할 GIF 파일의 경로.
    :param duration_ms: 각 프레임이 표시될 시간 (밀리초, 기본값 100ms).
    :param loop: 애니메이션 반복 횟수. 0은 무한 반복 (기본값 0).
    """
    
    if not input_png_paths:
        print("오류: 입력 PNG 파일 목록이 비어 있습니다.")
        return
        
    # 첫 번째 이미지를 로드합니다.
    try:
        frames = []
        first_frame = Image.open(input_png_paths[0])
        
        # 나머지 프레임들을 로드합니다.
        for path in input_png_paths[1:]:
            frames.append(Image.open(path))

    except FileNotFoundError:
        print(f"오류: 파일 경로를 찾을 수 없습니다. (경로: {path} 또는 {input_png_paths[0]})")
        return
    except Exception as e:
        print(f"이미지 로드 중 오류가 발생했습니다: {e}")
        return

    # 첫 번째 이미지와 나머지 이미지 목록을 사용하여 GIF를 저장합니다.
    try:
        first_frame.save(
            output_gif_path,
            format="GIF",
            append_images=frames,         # 첫 번째 프레임 뒤에 붙일 이미지들
            save_all=True,                # 모든 프레임을 저장하도록 설정
            duration=duration_ms,         # 프레임 간 지연 시간 (밀리초)
            loop=loop                     # 반복 횟수 (0: 무한 반복)
        )
        print(f"✅ Animated GIF 생성 성공: {output_gif_path} (프레임 수: {len(input_png_paths)}, 지연: {duration_ms}ms)")

    except Exception as e:
        print(f"GIF 저장 중 오류가 발생했습니다: {e}")

# --- 사용 예시 ---

# 주의: 이 코드를 실행하기 전에 'frame1.png', 'frame2.png', 'frame3.png' 등의 
#       실제 PNG 파일들이 같은 폴더에 준비되어 있어야 합니다.

INPUT_FRAMES = ["assets\iccdw_dot.png","assets\ccdw_dot.png","assets\clongcdw_dot.png"]
OUTPUT_FILE = "assets\\animation.gif"
FRAME_DURATION = 200 # 200ms = 0.2초마다 프레임 변경

# 예시 파일들이 실제 존재한다고 가정하고 함수 호출
# create_animated_gif(INPUT_FRAMES, OUTPUT_FILE, duration_ms=FRAME_DURATION)

# 💡 참고: 테스트용 더미 파일이 없다면 아래 코드를 실행하지 마세요!
# 만약 테스트를 위해 더미 파일이 필요하다면 별도로 요청해주세요.