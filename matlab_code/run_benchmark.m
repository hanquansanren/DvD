% add LD path
% https://people.csail.mit.edu/celiu/SIFTflow/
% change the path to your SIFTflow folder
addpath(genpath('~/SIFTflow'));
% GT images folder, for exampe: ~/data/docunet_benchmark/scan/
gtdir = './scan/';
% Unwarped image folder
% imdir = './115_1111/';
% imdir = './mix3/';
% imdir = './DocTr';
% imdir = './paperedge_result';
% imdir = './DocGeoTr';
% imdir = './144';
% imdir = './Marior';
% imdir = './RDGR';
% imdir = './crop';
% imdir = './dewarpnet';
% imdir = './DDDF';
% imdir = './inv3d/unwarped_';
% imdir = './UVDoc';
% imdir = './FTA';
%imdir = './0128_13';
%imdir = './0204_2';
%imdir = './0204_2_re';
imdir = './0204_3';
%imdir = './0204_3_re';

tarea=598400;
% res = zeros(64, 4);
res = cell(64, 1);
parpool(8);
parfor k = 1 : 64
    disp(k);
    rimg = imread(sprintf('%s/%d.png', gtdir, k));

    t = zeros(2, 5);
    for m = 1 : 2
        try
%             ximg = imread(sprintf('%s/%d_%d copy_rec.png', imdir, k, m));
            ximg = imread(sprintf('%s/warped_%d_%d copy.png', imdir, k, m));
%             ximg = imread(sprintf('%s/%d_%d.png', imdir, k, m));
%             ximg = imread(sprintf('%s/%d_%d.png', imdir, k, m));
            [ms, ld] = evalUnwarp(ximg, rimg);
            [~, relres] = evalAlignedUnwarp(ximg, rimg);
            t(m, :) = [k, m, relres, ms, ld];
        catch ME
            disp(ME.message)
            t(m, :) = [k, m, -1, -1, -1];
            % ms = 0;
            % ld = -1;
        end

    end
    res{k} = t;
end
res = cell2mat(res);
valres = res(res(:, 3) > 0, :);
avg = mean(valres, 1);
% avg = mean(res, 1);
res = cat(1, res, avg);

save(sprintf('%s/adres.txt', imdir), 'res', '-ascii');