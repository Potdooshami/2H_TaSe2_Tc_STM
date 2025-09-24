function [filePaths,isfiles] = checkFilesExsit(folder,fileNames)
for ifile = 1:length(fileNames)
    fileName = fileNames(ifile);
    filePath = fullfile(folder, fileName);
    filePaths(ifile) = filePath;
    isfiles(ifile) = isfile(filePath);
    if isfile(filePath)
        disp(['File exists: ', filePath]);
    else
        disp('File does not exist.');
    end
end
end