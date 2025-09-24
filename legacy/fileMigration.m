function fileMigration(folderNew,fileNameConfirmed,fileNameConfirmed_)
for ifile = 1:length(fileNameConfirmed)
    sourceFile = fileNameConfirmed(ifile);
    sourceFile_ = fileNameConfirmed_(ifile)
    destinationFile = fullfile(folderNew, sourceFile_);
    copyfile(sourceFile, destinationFile);
end
end