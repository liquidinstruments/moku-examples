const _ = require('lodash');
const fs = require('fs');
const glob = require('glob');

const subDirExpected = [
    { name: 'edl', display: 'Datalogger' },
    { name: 'eos', display: 'Oscilloscope' }
];

const isDirSupported = function(name) {
    return subDirExpected.findIndex(e => e.name == name) >= 0;
};

const getDisplayForDir = function(name) {
    return subDirExpected.find(e => e.name == name).display;
};

const getDirContent = function(path, isRoot = false) {
    files = glob.sync(path + '/*.md')
        .map(path => {
            f_name = path.split('/').pop().replace(".md", "")

            // Remove "README", making it the de facto index page
            if (path.endsWith('README.md')) {
                path = path.slice(0, -9);
                f_name = "Overview"
            }

            // "getters" should be "Getters"
            f_name = (f_name == "getters") ? "Getters" : f_name;

            path = path.replace("docs/", '');
            path = path.replace(".md", '');
            return {
                path,
                f_name
            };
        })

    var sortedfiles = _.sortBy(files, ['path'])
        .map(file => [file.path, file.f_name]);

    return (isRoot) ? sortedfiles : {
        'title': getDisplayForDir(path.split('/').pop()),
        'children': sortedfiles
    };

};

const getChildren = function(title, lookup_path) {
    complete_path = 'docs/reference/' + lookup_path;

    sub_directories = fs.readdirSync(complete_path, { withFileTypes: true })
        .filter(dirent => dirent.isDirectory() && isDirSupported(dirent.name))
        .map(dirent => complete_path + '/' + dirent.name);

    let groups = [complete_path].concat(sub_directories);

    data = groups.map(g => {
        return getDirContent(g, complete_path == g)
    });

    return {
        title: title,
        collapsable: true,
        children: _.concat(...data)
    };

};

module.exports = {
    getChildren,
};