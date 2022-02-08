function dataPreprocess()
    
    delete ./input/train/*.pgm

    % Get list of all subfolders.
    allSubFolders = genpath('.\input\all');
    
    % Parse into a cell array.
    remain = allSubFolders;
    
    listOfFolderNames = {};
    
    while true
      [singleSubFolder, remain] = strtok(remain, ';');
      if isempty(singleSubFolder)
        break;
      end
      listOfFolderNames = [listOfFolderNames singleSubFolder];
    end
    numberOfFolders = length(listOfFolderNames);
    
    countTrain = 1;
    countTest = 1;
    
    % Process all pgm files in those folders
    for k = 1 : numberOfFolders
        
      % Get this folder and print it out.
      thisFolder = listOfFolderNames{k};
      
      folderEnd = str2num(char(thisFolder(14:end)));
      
      sDirectory = sprintf('.\\input\\test\\s%d', folderEnd);
      
      if not(isfolder(sDirectory))
        mkdir(sDirectory);
        rmdir .\input\test\s
      end    
      

      % Get filenames of all pgm files.
      filePattern = sprintf('%s/*.pgm', thisFolder);
      baseFileNames = dir(filePattern);
      numberOfFiles = length(baseFileNames);
           
      % Now we have a list of all text files in this folder.
      if numberOfFiles >= 1
                          
        %get random indicies within the directory
        index = randperm(numberOfFiles, 10);
     
        % Go through all those text files.
        for f = 1 : length(index)
          pgmFileName = baseFileNames(index(f)).name;
          fullFileName = fullfile(thisFolder, pgmFileName);
          
          if(f < 9)
          
              filename = sprintf('.\\input\\train\\%d.pgm', countTrain);

              file = char(filename);

              A = imread(fullFileName);

              imwrite(A, file);

              countTrain = countTrain + 1;
              
          else
              
              directory = '.\\input\\test\\s%d\\%d.pgm';
            
              filename = sprintf(directory, folderEnd, countTest);

              file = char(filename);

              A = imread(fullFileName);

              imwrite(A, file);

              countTest = countTest + 1;
          
          end
        end
          countTest = 1;
      else
      end
    end

end

