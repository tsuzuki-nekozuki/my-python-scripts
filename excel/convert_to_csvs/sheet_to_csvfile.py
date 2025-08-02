import csv

from pathlib import Path

import openpyxl


def excel2csv(infile: str | Path, outdir: str | Path = '.'):
    """
    Create one csv file for each sheet in the workbook.

    Args:
        infile (str | Path: Input Excel file
        outdir (str | Path, optional): Output directory. Defaults to '.'.

    Raises:
        ValueError: If input Excel file does not exist.
        ValueError: If input file is not an Excel file.
    """
    infile = Path(infile)
    if not infile.exists():
        raise ValueError('"{}" does not exist.'.format(infile))
    if infile.suffix != '.xlsx':
        raise ValueError('"{}" is not a Excel file.'.format(infile))

    wb = openpyxl.load_workbook(str(infile))

    if not outdir.exists():
        outdir.mkdir(parents=True)

    for iws in wb.sheetnames:
        ws = wb[iws]
        if len(ws.merged_cells.ranges) > 0:
            print('Skip sheet "{}" because merged cells are detected.'
                  .format(iws))
            continue
        fout = outdir / '{}.csv'.format(iws)
        with open(str(fout), newline='', mode='w') as csvfile:
            writer = csv.writer(csvfile)
            for row in ws.rows:
                writer.writerow([cell.value for cell in row])
    wb.close()


if __name__ == '__main__':
    input = Path('iris_dataset.xlsx')
    outputs = Path('out_example')
    excel2csv(input, outputs)
