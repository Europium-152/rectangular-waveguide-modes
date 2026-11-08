import datetime
def write_csv(path, header_lines, rows, columns=('curve', 'x', 'y', 'uncertainty', 'x_uncertainty')):
    with open(path, 'w') as f:
        for h in header_lines:
            f.write('# ' + h + '\n')
        f.write(','.join(columns) + '\n')
        for r in rows:
            out = []
            for v in r:
                if isinstance(v, str):
                    out.append('"%s"' % v if ',' in v else v)
                elif isinstance(v, int):
                    out.append(str(v))
                else:
                    out.append('%.5g' % v)
            f.write(','.join(out) + '\n')
