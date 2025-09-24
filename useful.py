import matplotlib.pyplot as plt

def tickoff(ax = None):
  """
  Remove ticks from the current axes.
  """
    
  if ax is None:
    ax = plt.gca()
  
  ax.set_xticks([])
  ax.set_yticks([])

def gci():
    """
    Get the current image object.

    Returns
    -------
    matplotlib.image.AxesImage
        The most recently added image object on the current axes.
    """
    import matplotlib.pyplot as plt
    return plt.gca().images[-1]

def change_image_range(image_object, vmin=None, vmax=None):
    """
    Change the vmin and vmax of a matplotlib image object.

    Parameters
    ----------
    image_object : matplotlib.image.AxesImage
        The image object to modify.
    vmin : float, optional
        The new minimum value for the color mapping. If None, the current vmin is kept.
    vmax : float, optional
        The new maximum value for the color mapping. If None, the current vmax is kept.
    """
    if vmin is not None:
        image_object.set_clim(vmin=vmin)
    if vmax is not None:
        image_object.set_clim(vmax=vmax)

def auto_clim(data, method='percentile', lower=1, upper=99):    
    """
    data: 2D array
    method: 'percentile' or 'std'
    """
    import numpy as np
    if method == 'percentile':
        vmin, vmax = np.percentile(data, [lower, upper])
    elif method == 'std':
        mean = np.mean(data)
        std = np.std(data)
        vmin, vmax = mean - std, mean + std
    else:
        raise ValueError("method must be 'percentile' or 'std'")
    return vmin, vmax

def auto199():
  import matplotlib.pyplot as plt

  """
  Automatically adjust the color limits of the current image to the 1% and 99%
  """
  ic = gci()
  foo = ic.get_array()
  aclim =  auto_clim(foo, method='percentile', lower=1, upper=99)
  change_image_range(ic, aclim[0], aclim[1])


def plot_result(image, background):
    """
    Plot the original image, background, and the result of image - background side by side.
    """

    fig, ax = plt.subplots(nrows=1, ncols=3)

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original image')
    ax[0].axis('off')

    ax[1].imshow(background, cmap='gray')
    ax[1].set_title('Background')
    ax[1].axis('off')

    ax[2].imshow(image - background, cmap='gray')
    ax[2].set_title('Result')
    ax[2].axis('off')

    fig.tight_layout()
    return fig,ax

def subtabs(figs, titles=None):
    import ipywidgets as widgets
    from IPython.display import display
    tabs = widgets.Tab()
    tabs.children = [widgets.Output() for _ in figs]
    for i, fig in enumerate(figs):
        with tabs.children[i]:
            display(fig)
    if titles is not None:
        for i, title in enumerate(titles):
            tabs.set_title(i, title)
    return tabs

def subtabFigSr(figsSr):
    return subtabs(figsSr.to_list(),list(figsSr.index))


def subtabSr(arrSr):
    '''
    arrSr: pd.Series of matplotlib.figure.Figure
    '''
    import matplotlib.pyplot as plt
    def figimshow(arr):        
        fig = plt.figure(figsize=(20,20))
        plt.imshow(arr,cmap='afmhot')
        try:
            auto199()
        except:
            pass
        # tickoff()
        plt.close(fig)        
        return fig
    figs = list(map(figimshow,list(arrSr.values)))
    titles = arrSr.index.to_list()
    tabs = subtabs(figs,titles)
    return tabs



from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
import numpy as np
def color_histogram(arr, clim=None, nbins=200, cmap='gray',ax = None):
    """
    1D 배열 데이터에 대한 컬러 히스토그램을 생성하고 표시합니다.

    Args:
        arr (np.ndarray): 입력 1차원 NumPy 배열.
        clim (tuple, optional): 컬러맵의 범위를 지정하는 (최소, 최대) 튜플.
                                 기본값은 배열의 최소값과 최대값입니다.
        nbins (int, optional): 히스토그램의 빈(bin) 개수. 기본값은 200입니다.
        cmap (str or Colormap, optional): 사용할 Matplotlib 컬러맵.
                                            문자열 (예: 'viridis') 또는
                                            커스텀 컬러맵 객체를 사용할 수 있습니다.
                                            기본값은 'gray'입니다.
    """
    # 입력 데이터가 1차원인지 확인
    if arr.ndim != 1:
        raise ValueError("입력 배열은 반드시 1차원이어야 합니다.")

    # clim의 기본값 설정
    if clim is None:
        clim = (arr.min(), arr.max())

    # 히스토그램 계산
    n, bins = np.histogram(arr, bins=nbins, range=clim)

    # 각 bin의 중간값 계산
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # 컬러맵 가져오기
    colormap = get_cmap(cmap)

    # bin의 중간값을 0과 1 사이로 정규화
    norm = Normalize(vmin=clim[0], vmax=clim[1])
    normalized_centers = norm(bin_centers)

    # 정규화된 값에 따라 색상 결정
    colors = colormap(normalized_centers)

    # 히스토그램 그리기
    # fig, ax = plt.subplots()
    if ax is None:
        ax = plt.gca()
    ax.bar(bin_centers, n, width=(bins[1] - bins[0]), color=colors)

    # 축과 제목 설정
    ax.set_xlim(clim)
    ax.set_yticks([])
    # ax.set_xlabel('Value')
    # ax.set_ylabel('Frequency')
    # ax.set_title('Color Histogram')
    