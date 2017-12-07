import ConfigParser


# reader properties according to (section, key)
def parse_args(section, key):
    cf = ConfigParser.ConfigParser()
    #cf.read("D:/Normal-Software/pycharm/MLGame/properties.conf")
    cf.read("../properties.conf")
    # return all sections
    secs = cf.sections()
    # print "sections:", secs

    # global section
    return cf.get(section, key)


def base_path():
    return parse_args('global', 'project_path')


def separator():
    return parse_args('global', 'path_separator')


if __name__ == '__main__':
    project_path = parse_args("global", "project_path")
    print(project_path)
    print (separator())
