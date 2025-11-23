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
    


import matplotlib.colors as mcolors
import matplotlib.cm as cm


def add_colorbar_to_figure(f, cmap, clim, xxyy):
    """
    Figure 객체에 지정된 위치와 속성으로 컬러바를 추가합니다.

    <매개변수>
    f (plt.figure): 컬러바를 추가할 Figure 객체
    cmap (str or Colormap): 적용할 컬러맵
    clim (iterable): 컬러바의 최솟값과 최댓값을 담은 리스트 또는 튜플 (예: [0, 1])
    xxyy (iterable): Figure 좌표계(0~1) 기준 컬러바 위치 [left, bottom, width, height]

    <반환값>
    matplotlib.colorbar.Colorbar: 생성된 컬러바 객체
    """
    # 컬러바를 그릴 새로운 축(Axes)을 Figure에 추가
    cax = f.add_axes(xxyy)
    
    # 정규화(Normalization) 객체 생성
    norm = mcolors.Normalize(vmin=clim[0], vmax=clim[1])
    
    # 컬러맵과 정규화 정보를 담는 ScalarMappable 객체 생성
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    
    # 지정된 축에 컬러바 생성
    cb = f.colorbar(mappable, cax=cax, orientation='horizontal' if xxyy[2] > xxyy[3] else 'vertical')
    
    return cb

def add_colorbar_for_image(img, xxyy):
    """
    Image 객체의 속성(cmap, clim)을 이용해 Figure에 컬러바를 추가합니다.

    <매개변수>
    img (matplotlib.image.AxesImage): `plt.imshow` 등으로 생성된 이미지 객체
    xxyy (iterable): Figure 좌표계(0~1) 기준 컬러바 위치 [left, bottom, width, height]

    <반환값>
    matplotlib.colorbar.Colorbar: 생성된 컬러바 객체
    """
    # 이미지 객체로부터 Figure 객체를 가져옴
    fig = img.axes.figure
    
    # 컬러바를 그릴 새로운 축(Axes)을 Figure에 추가
    cax = fig.add_axes(xxyy)
    
    # 이미지 객체를 직접 사용하여 컬러바 생성
    # cmap과 clim 정보가 자동으로 이미지로부터 전달됨
    cb = fig.colorbar(img, cax=cax, orientation='horizontal' if xxyy[2] > xxyy[3] else 'vertical')
    
    return cb


import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.axes
import matplotlib.image
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np

def add_colorbar(artist, *, xxyy=None, cmap=None, clim=None):
    """
    Figure, Axes, Image 객체에 컬러바를 추가합니다.
    cmap, clim 정보가 주어지지 않으면 artist에서 자동으로 추론합니다.

    <매개변수>
    artist (Figure, Axes, or AxesImage): 컬러바를 추가할 대상 객체
    xxyy (iterable, optional): Figure 좌표계 기준 컬러바 위치 [l, b, w, h].
                                기본값: [0.8, 0.95, 0.1, 0.02]
    cmap (str or Colormap, optional): 적용할 컬러맵. 없으면 artist에서 추론.
    clim (iterable, optional): 컬러바의 최솟값/최댓값. 없으면 artist에서 추론.

    <반환값>
    matplotlib.colorbar.Colorbar: 생성된 컬러바 객체
    """
    # 1. artist 타입에 따라 figure와 image 정보를 찾습니다.
    if isinstance(artist, matplotlib.image.AxesImage):
        target_image = artist
        f = target_image.get_figure()
    elif isinstance(artist, matplotlib.axes.Axes):
        if not artist.images:
            raise ValueError("입력된 Axes 객체에 이미지가 없습니다.")
        target_image = artist.images[-1]
        f = artist.get_figure()
    elif isinstance(artist, matplotlib.figure.Figure):
        if not artist.axes:
            raise ValueError("입력된 Figure 객체에 축(Axes)이 없습니다.")
        ax = artist.axes[-1]
        if not ax.images:
            raise ValueError("Figure의 마지막 Axes에 이미지가 없습니다.")
        target_image = ax.images[-1]
        f = artist
    else:
        raise TypeError("artist는 Figure, Axes, AxesImage 타입이어야 합니다.")

    # 2. cmap과 clim이 주어지지 않으면 image에서 추론합니다.
    if cmap is None:
        cmap = target_image.get_cmap()
    if clim is None:
        clim = target_image.get_clim()

    # 3. xxyy의 기본값을 설정합니다.
    if xxyy is None:
        xxyy = [0.8, 0.95, 0.1, 0.02]

    # 4. 컬러바를 생성합니다. (기존 로직 활용)
    cax = f.add_axes(xxyy)
    norm = mcolors.Normalize(vmin=clim[0], vmax=clim[1])
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    orientation = 'horizontal' if xxyy[2] > xxyy[3] else 'vertical'
    cb = f.colorbar(mappable, cax=cax, orientation=orientation)
    
    return cb



def subplotss(nm_rows, nm_cols, nm_sup='sup', figsize=(10, 6), **kwargs):
    """
    Wrapper function for plt.subplots
    
    Parameters:
    -----------
    nm_rows : list
        List of ylabel for each row
    nm_cols : list  
        List of title for each column
    nm_sup : str, default='sup'
        Suptitle for the entire figure
    figsize : tuple, default=(10, 6)
        Figure size
    **kwargs : 
        Additional arguments to pass to plt.subplots (sharex, sharey, subplot_kw, etc.)
    
    Returns:
    --------
    fig, axs : matplotlib figure and axes objects
    """
    # Create subplots using lengths of nm_rows and nm_cols
    nrows, ncols = len(nm_rows), len(nm_cols)
    
    # Pass all arguments to plt.subplots
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    
    # Set suptitle
    plt.suptitle(nm_sup)
    
    # Convert to 2D array if axs is 1D (for single row/column cases)
    if nrows == 1:
        axs = axs.reshape(1, -1)
    elif ncols == 1:
        axs = axs.reshape(-1, 1)
    elif nrows == 1 and ncols == 1:
        axs = np.array([[axs]])
    
    # Set ylabel for each row (only on the first column)
    for i, row_label in enumerate(nm_rows):
        axs[i, 0].set_ylabel(row_label)
    
    # Set title for each column (only on the first row)
    for j, col_label in enumerate(nm_cols):
        axs[0, j].set_title(col_label)
    
    return fig, axs


def _prepare_broadcastable_arg(arg, target_shape):
    """(이전과 동일한 헬퍼 함수)"""
    m, n = target_shape
    np_arg = np.array(arg, dtype=object)

    if np_arg.ndim == 1:
        if np_arg.shape[0] == n:
            np_arg = np_arg.reshape(1, n)
        elif np_arg.shape[0] == m:
            np_arg = np_arg.reshape(m, 1)

    try:
        return np.broadcast_to(np_arg, target_shape)
    except ValueError:
        print(f"Warning: The provided argument with shape {np.shape(arg)} could not be broadcast to {target_shape}. Treating as a single element.")
        arg_as_scalar = np.empty((1, 1), dtype=object)
        arg_as_scalar[0, 0] = np_arg

        # arg_as_scalar = np.array([[arg]], dtype=object)
        print(arg_as_scalar.shape)
        return np.broadcast_to(arg_as_scalar, target_shape)

def broad_plot(axs, dts, draws):
    """
    여러 axes에 데이터와 그리기 함수를 브로드캐스팅하여 그림을 그립니다. (수정된 버전)
    """
    if not isinstance(axs, np.ndarray):
        axs = np.array(axs)
    
    target_shape = axs.shape
    broadcasted_dts = _prepare_broadcastable_arg(dts, target_shape)
    broadcasted_draws = _prepare_broadcastable_arg(draws, target_shape)

    for i, j in np.ndindex(target_shape):
        ax = axs[i, j]
        dt = broadcasted_dts[i, j]
        draw_func = broadcasted_draws[i, j]
        
        # 💡 변경된 부분: ax를 직접 인자로 전달
        draw_func(ax, dt)
def fullax(ax=None):
    if ax is None:
        ax = plt.gca()
    ax.set_position([0,0,1,1])
    ax.set_xticks([])
    cax = f.add_axes(xxyy)
    
    # 정규화(Normalization) 객체 생성
    norm = mcolors.Normalize(vmin=clim[0], vmax=clim[1])
    
    # 컬러맵과 정규화 정보를 담는 ScalarMappable 객체 생성
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    
    # 지정된 축에 컬러바 생성
    cb = f.colorbar(mappable, cax=cax, orientation='horizontal' if xxyy[2] > xxyy[3] else 'vertical')
    
    return cb

def add_colorbar_for_image(img, xxyy):
    """
    Image 객체의 속성(cmap, clim)을 이용해 Figure에 컬러바를 추가합니다.

    <매개변수>
    img (matplotlib.image.AxesImage): `plt.imshow` 등으로 생성된 이미지 객체
    xxyy (iterable): Figure 좌표계(0~1) 기준 컬러바 위치 [left, bottom, width, height]

    <반환값>
    matplotlib.colorbar.Colorbar: 생성된 컬러바 객체
    """
    # 이미지 객체로부터 Figure 객체를 가져옴
    fig = img.axes.figure
    
    # 컬러바를 그릴 새로운 축(Axes)을 Figure에 추가
    cax = fig.add_axes(xxyy)
    
    # 이미지 객체를 직접 사용하여 컬러바 생성
    # cmap과 clim 정보가 자동으로 이미지로부터 전달됨
    cb = fig.colorbar(img, cax=cax, orientation='horizontal' if xxyy[2] > xxyy[3] else 'vertical')
    
    return cb


import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.axes
import matplotlib.image
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np

def add_colorbar(artist, *, xxyy=None, cmap=None, clim=None):
    """
    Figure, Axes, Image 객체에 컬러바를 추가합니다.
    cmap, clim 정보가 주어지지 않으면 artist에서 자동으로 추론합니다.

    <매개변수>
    artist (Figure, Axes, or AxesImage): 컬러바를 추가할 대상 객체
    xxyy (iterable, optional): Figure 좌표계 기준 컬러바 위치 [l, b, w, h].
                                기본값: [0.8, 0.95, 0.1, 0.02]
    cmap (str or Colormap, optional): 적용할 컬러맵. 없으면 artist에서 추론.
    clim (iterable, optional): 컬러바의 최솟값/최댓값. 없으면 artist에서 추론.

    <반환값>
    matplotlib.colorbar.Colorbar: 생성된 컬러바 객체
    """
    # 1. artist 타입에 따라 figure와 image 정보를 찾습니다.
    if isinstance(artist, matplotlib.image.AxesImage):
        target_image = artist
        f = target_image.get_figure()
    elif isinstance(artist, matplotlib.axes.Axes):
        if not artist.images:
            raise ValueError("입력된 Axes 객체에 이미지가 없습니다.")
        target_image = artist.images[-1]
        f = artist.get_figure()
    elif isinstance(artist, matplotlib.figure.Figure):
        if not artist.axes:
            raise ValueError("입력된 Figure 객체에 축(Axes)이 없습니다.")
        ax = artist.axes[-1]
        if not ax.images:
            raise ValueError("Figure의 마지막 Axes에 이미지가 없습니다.")
        target_image = ax.images[-1]
        f = artist
    else:
        raise TypeError("artist는 Figure, Axes, AxesImage 타입이어야 합니다.")

    # 2. cmap과 clim이 주어지지 않으면 image에서 추론합니다.
    if cmap is None:
        cmap = target_image.get_cmap()
    if clim is None:
        clim = target_image.get_clim()

    # 3. xxyy의 기본값을 설정합니다.
    if xxyy is None:
        xxyy = [0.8, 0.95, 0.1, 0.02]

    # 4. 컬러바를 생성합니다. (기존 로직 활용)
    cax = f.add_axes(xxyy)
    norm = mcolors.Normalize(vmin=clim[0], vmax=clim[1])
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    orientation = 'horizontal' if xxyy[2] > xxyy[3] else 'vertical'
    cb = f.colorbar(mappable, cax=cax, orientation=orientation)
    
    return cb



def subplotss(nm_rows, nm_cols, nm_sup='sup', figsize=(10, 6), **kwargs):
    """
    Wrapper function for plt.subplots
    
    Parameters:
    -----------
    nm_rows : list
        List of ylabel for each row
    nm_cols : list  
        List of title for each column
    nm_sup : str, default='sup'
        Suptitle for the entire figure
    figsize : tuple, default=(10, 6)
        Figure size
    **kwargs : 
        Additional arguments to pass to plt.subplots (sharex, sharey, subplot_kw, etc.)
    
    Returns:
    --------
    fig, axs : matplotlib figure and axes objects
    """
    # Create subplots using lengths of nm_rows and nm_cols
    nrows, ncols = len(nm_rows), len(nm_cols)
    
    # Pass all arguments to plt.subplots
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    
    # Set suptitle
    plt.suptitle(nm_sup)
    
    # Convert to 2D array if axs is 1D (for single row/column cases)
    if nrows == 1:
        axs = axs.reshape(1, -1)
    elif ncols == 1:
        axs = axs.reshape(-1, 1)
    elif nrows == 1 and ncols == 1:
        axs = np.array([[axs]])
    
    # Set ylabel for each row (only on the first column)
    for i, row_label in enumerate(nm_rows):
        axs[i, 0].set_ylabel(row_label)
    
    # Set title for each column (only on the first row)
    for j, col_label in enumerate(nm_cols):
        axs[0, j].set_title(col_label)
    
    return fig, axs


def _prepare_broadcastable_arg(arg, target_shape):
    """(이전과 동일한 헬퍼 함수)"""
    m, n = target_shape
    np_arg = np.array(arg, dtype=object)

    if np_arg.ndim == 1:
        if np_arg.shape[0] == n:
            np_arg = np_arg.reshape(1, n)
        elif np_arg.shape[0] == m:
            np_arg = np_arg.reshape(m, 1)

    try:
        return np.broadcast_to(np_arg, target_shape)
    except ValueError:
        print(f"Warning: The provided argument with shape {np.shape(arg)} could not be broadcast to {target_shape}. Treating as a single element.")
        arg_as_scalar = np.empty((1, 1), dtype=object)
        arg_as_scalar[0, 0] = np_arg

        # arg_as_scalar = np.array([[arg]], dtype=object)
        print(arg_as_scalar.shape)
        return np.broadcast_to(arg_as_scalar, target_shape)

def broad_plot(axs, dts, draws):
    """
    여러 axes에 데이터와 그리기 함수를 브로드캐스팅하여 그림을 그립니다. (수정된 버전)
    """
    if not isinstance(axs, np.ndarray):
        axs = np.array(axs)
    
    target_shape = axs.shape
    broadcasted_dts = _prepare_broadcastable_arg(dts, target_shape)
    broadcasted_draws = _prepare_broadcastable_arg(draws, target_shape)

    for i, j in np.ndindex(target_shape):
        ax = axs[i, j]
        dt = broadcasted_dts[i, j]
        draw_func = broadcasted_draws[i, j]
        
        # 💡 변경된 부분: ax를 직접 인자로 전달
        draw_func(ax, dt)
def fullax(ax=None):
    if ax is None:
        ax = plt.gca()
    ax.set_position([0,0,1,1])
    ax.set_xticks([])
    cax = f.add_axes(xxyy)
    
    # 정규화(Normalization) 객체 생성
    norm = mcolors.Normalize(vmin=clim[0], vmax=clim[1])
    
    # 컬러맵과 정규화 정보를 담는 ScalarMappable 객체 생성
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    
    # 지정된 축에 컬러바 생성
    cb = f.colorbar(mappable, cax=cax, orientation='horizontal' if xxyy[2] > xxyy[3] else 'vertical')
    
    return cb

def add_colorbar_for_image(img, xxyy):
    """
    Image 객체의 속성(cmap, clim)을 이용해 Figure에 컬러바를 추가합니다.

    <매개변수>
    img (matplotlib.image.AxesImage): `plt.imshow` 등으로 생성된 이미지 객체
    xxyy (iterable): Figure 좌표계(0~1) 기준 컬러바 위치 [left, bottom, width, height]

    <반환값>
    matplotlib.colorbar.Colorbar: 생성된 컬러바 객체
    """
    # 이미지 객체로부터 Figure 객체를 가져옴
    fig = img.axes.figure
    
    # 컬러바를 그릴 새로운 축(Axes)을 Figure에 추가
    cax = fig.add_axes(xxyy)
    
    # 이미지 객체를 직접 사용하여 컬러바 생성
    # cmap과 clim 정보가 자동으로 이미지로부터 전달됨
    cb = fig.colorbar(img, cax=cax, orientation='horizontal' if xxyy[2] > xxyy[3] else 'vertical')
    
    return cb


import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.axes
import matplotlib.image
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np

def add_colorbar(artist, *, xxyy=None, cmap=None, clim=None):
    """
    Figure, Axes, Image 객체에 컬러바를 추가합니다.
    cmap, clim 정보가 주어지지 않으면 artist에서 자동으로 추론합니다.

    <매개변수>
    artist (Figure, Axes, or AxesImage): 컬러바를 추가할 대상 객체
    xxyy (iterable, optional): Figure 좌표계 기준 컬러바 위치 [l, b, w, h].
                                기본값: [0.8, 0.95, 0.1, 0.02]
    cmap (str or Colormap, optional): 적용할 컬러맵. 없으면 artist에서 추론.
    clim (iterable, optional): 컬러바의 최솟값/최댓값. 없으면 artist에서 추론.

    <반환값>
    matplotlib.colorbar.Colorbar: 생성된 컬러바 객체
    """
    # 1. artist 타입에 따라 figure와 image 정보를 찾습니다.
    if isinstance(artist, matplotlib.image.AxesImage):
        target_image = artist
        f = target_image.get_figure()
    elif isinstance(artist, matplotlib.axes.Axes):
        if not artist.images:
            raise ValueError("입력된 Axes 객체에 이미지가 없습니다.")
        target_image = artist.images[-1]
        f = artist.get_figure()
    elif isinstance(artist, matplotlib.figure.Figure):
        if not artist.axes:
            raise ValueError("입력된 Figure 객체에 축(Axes)이 없습니다.")
        ax = artist.axes[-1]
        if not ax.images:
            raise ValueError("Figure의 마지막 Axes에 이미지가 없습니다.")
        target_image = ax.images[-1]
        f = artist
    else:
        raise TypeError("artist는 Figure, Axes, AxesImage 타입이어야 합니다.")

    # 2. cmap과 clim이 주어지지 않으면 image에서 추론합니다.
    if cmap is None:
        cmap = target_image.get_cmap()
    if clim is None:
        clim = target_image.get_clim()

    # 3. xxyy의 기본값을 설정합니다.
    if xxyy is None:
        xxyy = [0.8, 0.95, 0.1, 0.02]

    # 4. 컬러바를 생성합니다. (기존 로직 활용)
    cax = f.add_axes(xxyy)
    norm = mcolors.Normalize(vmin=clim[0], vmax=clim[1])
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    orientation = 'horizontal' if xxyy[2] > xxyy[3] else 'vertical'
    cb = f.colorbar(mappable, cax=cax, orientation=orientation)
    
    return cb



def subplotss(nm_rows, nm_cols, nm_sup='sup', figsize=(10, 6), **kwargs):
    """
    Wrapper function for plt.subplots
    
    Parameters:
    -----------
    nm_rows : list
        List of ylabel for each row
    nm_cols : list  
        List of title for each column
    nm_sup : str, default='sup'
        Suptitle for the entire figure
    figsize : tuple, default=(10, 6)
        Figure size
    **kwargs : 
        Additional arguments to pass to plt.subplots (sharex, sharey, subplot_kw, etc.)
    
    Returns:
    --------
    fig, axs : matplotlib figure and axes objects
    """
    # Create subplots using lengths of nm_rows and nm_cols
    nrows, ncols = len(nm_rows), len(nm_cols)
    
    # Pass all arguments to plt.subplots
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    
    # Set suptitle
    plt.suptitle(nm_sup)
    
    # Convert to 2D array if axs is 1D (for single row/column cases)
    if nrows == 1:
        axs = axs.reshape(1, -1)
    elif ncols == 1:
        axs = axs.reshape(-1, 1)
    elif nrows == 1 and ncols == 1:
        axs = np.array([[axs]])
    
    # Set ylabel for each row (only on the first column)
    for i, row_label in enumerate(nm_rows):
        axs[i, 0].set_ylabel(row_label)
    
    # Set title for each column (only on the first row)
    for j, col_label in enumerate(nm_cols):
        axs[0, j].set_title(col_label)
    
    return fig, axs


def _prepare_broadcastable_arg(arg, target_shape):
    """(이전과 동일한 헬퍼 함수)"""
    m, n = target_shape
    np_arg = np.array(arg, dtype=object)

    if np_arg.ndim == 1:
        if np_arg.shape[0] == n:
            np_arg = np_arg.reshape(1, n)
        elif np_arg.shape[0] == m:
            np_arg = np_arg.reshape(m, 1)

    try:
        return np.broadcast_to(np_arg, target_shape)
    except ValueError:
        print(f"Warning: The provided argument with shape {np.shape(arg)} could not be broadcast to {target_shape}. Treating as a single element.")
        arg_as_scalar = np.empty((1, 1), dtype=object)
        arg_as_scalar[0, 0] = np_arg

        # arg_as_scalar = np.array([[arg]], dtype=object)
        print(arg_as_scalar.shape)
        return np.broadcast_to(arg_as_scalar, target_shape)

def broad_plot(axs, dts, draws):
    """
    여러 axes에 데이터와 그리기 함수를 브로드캐스팅하여 그림을 그립니다. (수정된 버전)
    """
    if not isinstance(axs, np.ndarray):
        axs = np.array(axs)
    
    target_shape = axs.shape
    broadcasted_dts = _prepare_broadcastable_arg(dts, target_shape)
    broadcasted_draws = _prepare_broadcastable_arg(draws, target_shape)

    for i, j in np.ndindex(target_shape):
        ax = axs[i, j]
        dt = broadcasted_dts[i, j]
        draw_func = broadcasted_draws[i, j]
        
        # 💡 변경된 부분: ax를 직접 인자로 전달
        draw_func(ax, dt)
def fullax(ax=None):
    if ax is None:
        ax = plt.gca()
    ax.set_position([0,0,1,1])
    ax.set_xticks([])
    ax.set_yticks([])
def savepng(fig,fn):
    fig.savefig("assets/"+fn+".png",dpi=100,bbox_inches='tight',pad_inches=0)

import matplotlib.patches as patches
class CropWindow:
    """
    A class to define and manage a specific crop window (region of interest) for images or plots.
    
    It defines the region based on coordinates (x, y) and dimensions (width, height).
    Provides functionality to apply this region to Matplotlib Axes (set_xlim/ylim)
    or visualize it as a rectangle patch.
    """
    anchor_map = {
        'center': (0.5, 0.5), 'top': (0.5, 1.0), 'bottom': (0.5, 0.0),
        'left': (0.0, 0.5), 'right': (1.0, 0.5), 'top_left': (0.0, 1.0),
        'top_right': (1.0, 1.0), 'bottom_left': (0.0, 0.0), 'bottom_right': (1.0, 0.0),
    }
    def __init__(self):
        self._xy_01 = np.zeros((2,2))        
    def set_xlimylim(self,xlim,ylim):
        """
        Sets the region by directly specifying the range (min, max) for x and y axes.
        
        Args:
            xlim (tuple): (x_min, x_max)
            ylim (tuple): (y_min, y_max)
        """
        self._xy_01[0] = np.array(xlim)
        self._xy_01[1] = np.array(ylim)
        # self._xy_01[1] = np.array(self._xy_01[1])[::-1]
    def set_yflip(self):
        self._xy_01[1] = np.array(self._xy_01[1])[::-1]
    def set_by_anchor(self,xy,wh,anchor='center'):
        """
        Sets the region centered around a specific anchor point.
        
        Args:
            xy (tuple): Anchor point coordinates (x, y)
            wh (tuple): Size of the region (width, height)
            anchor (str): Position of the anchor ('center', 'bottom_left', etc.). Default is 'center'.
        """
        print(xy)
        xy = np.array(xy)
        print(wh)

        wh = np.array(wh) +np.array([0,0])
        # print()
        anchor = np.array(CropWindow.anchor_map[anchor])
        print(anchor)
        bl = xy - anchor * wh
        self._xy_01[:,0] = bl
        self._xy_01[:,1] = bl + wh
    @property
    def xlim(self):
        """Returns the x-axis range (min, max)."""
        return self._xy_01[0]
    @property
    def ylim(self):
        """Returns the y-axis range (min, max)."""
        return self._xy_01[1]
    @property
    def ylim_flip(self):
        """Returns the y-axis range (min, max)."""
        return self._xy_01[1][::-1]
    @property
    def wh(self):
        """Returns the width and height of the region."""
        return self._xy_01[:,1] - self._xy_01[:,0]
    @property
    def xy(self,anchor = 'bottom_left'):
        """
        Returns the coordinates of the specified anchor position.
        Default is 'bottom_left' (the bottom-left corner of the rectangle).
        """
        return self._xy_01[:,0] + self.anchor_map[anchor] * self.wh
    # @property
    def rect(self,**kwargs):
        """Returns a matplotlib.patches.Rectangle object representing the current region."""
        return patches.Rectangle(self.xy,self.wh[0],self.wh[1],fill=False,**kwargs)
    def ax_xylims(self,ax=None):
        """Applies the current region as the display range (xlim, ylim) to a Matplotlib Axes."""
        if ax is None:
            ax = plt.gca()
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim_flip)
    def ax_cropbox(self,ax=None,**kwargs):
        """Draws a rectangle box representing the current region on a Matplotlib Axes."""
        if ax is None:
            ax = plt.gca()
        ax.add_patch(self.rect(**kwargs))