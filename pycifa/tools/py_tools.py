import copy

def addStringInFilename(filename, string, prefix=True):
    # three independent options:
    # 1) is "string" prefix or postfix;
    # 2) does "filename" contain '/' symbol or not;
    # 3) does "filename" contain '.' symbol after last '/' or not
    # Therefore, we must consider 2^3=8 separated cases
    if not ('/' in filename):
        if prefix:
            return string + filename
        else:
            if not ('.' in filename):
                return filename + string
            else:
                tmp = filename[::-1].split('.', 1)
                tmp = tmp[1][::-1] + string + '.' + tmp[0][::-1]
                return tmp
    tmp = filename[::-1].split('/', 1)
    if prefix:
        return tmp[1][::-1] + '/' + string + tmp[0][::-1]
    else:
        if not ('.' in tmp[0]):
            return tmp[1][::-1] + '/' + tmp[0][::-1] + string
        else:
            tmp = tmp[0].split('.', 1) + tmp[1:]
            return tmp[2][::-1] + '/' + tmp[1][::-1] + string + tmp[0][::-1]


def pyHeadExtract(pyFname):
    funLines = []
    with open(pyFname, 'r') as f:
        lenBuf = 1
        while lenBuf > 0:
            tmp = f.readline()
            lenBuf = len(tmp)
            if ('def ' in tmp):
                funLines.append(tmp)
                while not tmp[:-1].endswith(':'):
                    tmp = f.readline()
                    funLines.append(tmp)
            if (' return ' in tmp):
                tmp2 = tmp.split(' return ')
                tmp2[0] = tmp2[0].replace(' ', '')
                if len(tmp2[0]) == 0:
                    funLines.append(tmp)
    headFname = addStringInFilename(pyFname, 'header_', prefix=True)
    with open(headFname, 'w+') as f:
        f.writelines(funLines)
    return 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
