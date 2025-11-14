from pathlib import Path
from typing import Iterable, Optional, Union
import argparse

import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_COLS = ("open", "high", "low", "close", "volume")


#Đọc csv dầu thô, vẽ histogram cho các cột số
def crude_oil_hist( 
    csv_path: Optional[Union[str, Path]] = None,
    cols: Optional[Iterable[str]] = None,
    bins: Union[int, str] = 50,
    show: bool = True,
    save_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:

    here = Path(__file__).parent

    # 1) Xác định đường dẫn CSV
    if csv_path is None:
        common_paths = [
            here / "Crude_Oil_Futures.csv",
            here.parent / "Crude_Oil_Futures.csv",
            here.parent.parent / "Crude_Oil_Futures.csv",
        ]
        for path in common_paths:
            if path.is_file():
                csv_path = path
                break
        else:
            raise FileNotFoundError("Không tìm thấy file Crude_Oil_Futures.csv ở vị trí phổ biến.")

    csv_path = Path(csv_path)
    print(f"[INFO] Đọc CSV từ: {csv_path.resolve()}")

    # 2) Đọc & chuẩn hoá tên cột
    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip() for c in df.columns]

    # 3) Ép kiểu số cho các cột cần thiết   
    use_cols = cols if cols is not None else DEFAULT_COLS
    for col in use_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            print(f"[WARNING] Cột '{col}' không tồn tại trong DataFrame.")

  #4) Tạo thư mục lưu ảnh nếu cần
    outdir = None
    if save_dir is not None:
        outdir = Path(save_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Ảnh sẽ được lưu vào: {outdir.resolve()}")

    for col in use_cols:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            continue

        plt.figure()
        plt.hist(series, bins=bins)
        plt.title(f"Phân phối {col}")
        plt.xlabel(col)
        plt.ylabel("Tần suất")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if outdir is not None:
            plt.savefig(outdir / f"hist_{col}.png", dpi=150)
            plt.close()

    # Nếu không lưu file thì mới cần show để khỏi pop nhiều cửa sổ khi đã lưu
    if show and outdir is None:
        plt.show()

    return df


def main():
    """CLI: cho phép chạy trực tiếp file này như một script."""
    parser = argparse.ArgumentParser(description="Vẽ histogram cho dữ liệu Crude Oil Futures.")
    parser.add_argument(
        "-c", "--csv", dest="csv_path", default=None,
        help="Đường dẫn CSV. Nếu bỏ trống, module sẽ tự tìm ở vị trí phổ biến."
    )
    parser.add_argument(
        "--cols", nargs="*", default=list(DEFAULT_COLS),
        help="Các cột số để vẽ (mặc định: open high low close volume)."
    )
    parser.add_argument(
        "--bins", default="50",
        help="Số bins (vd 50) hoặc 'auto'."
    )
    parser.add_argument(
        "--save-dir", default=None,
        help="Thư mục lưu ảnh PNG. Nếu không set, sẽ chỉ hiển thị (show)."
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Không mở cửa sổ biểu đồ (hữu ích khi chỉ muốn lưu ảnh)."
    )

    args = parser.parse_args()
    try:
        bins_val = int(args.bins)
    except ValueError:
        bins_val = args.bins  # cho phép 'auto'

    crude_oil_hist(
        csv_path=args.csv_path,
        cols=args.cols,
        bins=bins_val,
        show=not args.no_show,
        save_dir=args.save_dir,
    )
if __name__ == "__main__":
    main()
