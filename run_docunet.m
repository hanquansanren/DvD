% 添加 SIFTflow 工具箱路径（请根据实际情况修改路径）
addpath(genpath('~/SIFTflow'));
delete(gcp('nocreate'));   



% GT 图像文件夹，GT 图像文件名为 "1.png" ~ "64.png"
gtdir  = 'F:\evaluation_benchmark\dewarping_models\docscanner\scan';

% 矫正图像文件夹，矫正图像文件名格式为 x_a_b_k_m_geo.png

imdir = 'F:\evaluation_benchmark\dewarping_models\docscanner\DocScanner-L_DocUNet_rec-20250718T183829Z-1-001\DocScanner-L_DocUNet_rec';

tarea = 598400;


% 获取矫正图像文件夹中所有符合模式的文件
filePattern = fullfile(imdir, '*.png');
files = dir(filePattern);
nfiles = length(files);
fprintf('共找到 %d 个矫正图像文件\n', nfiles);


% 初始化结果矩阵，每行记录：[GT编号, idx, 对齐指标, MS指标, LD指标]
results = zeros(nfiles, 5);

% 开启 8 个并行工作者
parpool(8);

% 使用 parfor 遍历所有矫正图像文件
parfor i = 1:nfiles
    fileName = files(i).name;
    filePath = fullfile(imdir, fileName);
    
    % 使用正则表达式提取文件名中的数字
    % 文件名格式
    %tokens = regexp(fileName, '^(\d+)_([1-4])\.png$', 'tokens');
    %tokens = regexp(fileName, '^(\d+)_([1-4])_geo\.png$', 'tokens');
    %tokens = regexp(fileName, '^(\d+)_([1-4])_rec\.png$', 'tokens');
    %tokens = regexp(fileName, '^(\d+)_([1-4])_unwarp\.png$', 'tokens');
    tokens = regexp(fileName, '^(\d+)_([1-2]) copy_rec\.png$', 'tokens');

   
    if isempty(tokens)
        fprintf('文件名格式不匹配: %s\n', fileName);
        results(i,:) = [-1, -1, -1, -1, -1];
        continue;
    end
    tokens = tokens{1};  % 提取第一个匹配结果
    
    % 从文件名中提取参数，其中 k 为 GT 图像编号
    k = str2double(tokens{1});
    m = str2double(tokens{2});
    
    % 构造对应的 GT 图像文件名（假设 GT 图像为两位数字，例如 "01.png"）
    gtFileName = fullfile(gtdir, sprintf('%d.png', k));
    if exist(gtFileName, 'file') ~= 2
        fprintf('GT 图像不存在: %s\n', gtFileName);
        results(i,:) = [k, i, -1, -1, -1];
        continue;
    end
    rimg = imread(gtFileName);
    rimg = resize_keep_ar(rimg, tarea);
    
    % 读取矫正图像
    try
        ximg = imread(filePath);
        ximg = resize_keep_ar(ximg, tarea);
    catch ME
        fprintf('读取矫正图像出错 %s: %s\n', fileName, ME.message);
        results(i,:) = [k, i, -1, -1, -1];
        continue;
    end
    
    % 计算评估指标（调用外部函数 evalUnwarp 与 evalAlignedUnwarp）
    try
        [ms, ld] = evalUnwarp(ximg, rimg);
        [~, relres] = evalAlignedUnwarp(ximg, rimg);
    catch ME
        fprintf('计算评估指标出错 %s: %s\n', fileName, ME.message);
        results(i,:) = [k, i, -1, -1, -1];
        continue;
    end
    
    % 将本次计算结果保存到结果矩阵中
    % 记录：[GT编号, idx, 对齐指标, MS指标, LD指标]
    results(i,:) = [k, i, relres, ms, ld];
    
    fprintf('处理完成: %s\n', fileName);
end




% 可选：如果需要进一步筛选有效数据和计算均值，比如以对齐指标（第3列）> 0作为有效数据的判断条件
valres = results(results(:,3) > 0, :);
avg = mean(valres, 1);
results = cat(1, results, avg);

% 将结果保存为 ASCII 格式的文本文件，文件名为 results.txt，存放在矫正图像文件夹下
save(fullfile(imdir, 'adres.txt'), 'results', '-ascii');

fprintf('所有处理完成，结果保存在: %s\n', fullfile(imdir, 'adres.txt'));




% --- helper ---
function out = resize_keep_ar(img, targetArea)
    [h, w, ~] = size(img);
    scale = sqrt(targetArea / (h * w));
    newSize = round([h, w] * scale);
    % 使用双线性插值；对真值图像可用 nearest 保留像素级标注
    out = imresize(img, newSize, 'bilinear');
end
% ---------------