const fs = require('fs');
const path = require('path');

function getFilesSize(dirPath) {
  let totalSize = 0;
  const files = fs.readdirSync(dirPath);

  files.forEach((file) => {
    const filePath = path.join(dirPath, file);
    const stat = fs.statSync(filePath);

    if (stat.isFile() && (path.extname(filePath) === '.js' || path.extname(filePath) === '.vue')) {
      totalSize += stat.size;
    } else if (stat.isDirectory()) {
      totalSize += getFilesSize(filePath);
    }
  });

  return totalSize;
}

const dirPath = '/Users/nijunjie/WFE/NetEase/global-web-static/projects/global-web-static/src'; // 替换为你要统计的文件夹路径
const totalSize = getFilesSize(dirPath);

console.log(`Total size of JS and Vue files in ${dirPath}: ${totalSize} bytes`);
