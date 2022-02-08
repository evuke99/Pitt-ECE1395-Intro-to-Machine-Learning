function im_out = Segment_kmeans(im_in, K, iters, R)

    %convert them to doubles
    im = im2double(im_in);
    
    H = size(im,1);
    W = size(im,2);
    
    %downsample the images
    im = imresize(im, [100 100]);

    H1 = size(im,1);
    W1 = size(im,2);

    %convert 3D to 2D matrix with pixels as the rows and features as cols
    X = reshape(im, H1*W1, 3);

    %performm clustering
    [ids, means, ~] = kmeans_multiple(X, K, iters, R);

    %initialize temp X
    tempX = zeros(size(X));
    
    %loop over all pixels and assign its value to the mean value of the
    %pixel based on its cluster
    for pixel = 1:size(X,1)
        tempX(pixel, :) = means(ids(pixel), :);
    end
    
    %reshape tempX to [100 100 3] from [10000 3]
    im_out = reshape(tempX, H1, W1, 3);
    
    %resizes image back to original size
    im_out = imresize(im_out, [H W]);
    
    %multiply double array by 255 to get RGB values
    im_out = im_out*255;
    
    %convert double array to uint8 array and return the new image
    im_out = uint8(im_out);
    
end

