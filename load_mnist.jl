using Serialization

function load_mnist(;normalize=true, flatten=true, one_hot_label=false)

    # 引数パラメータ初期値
    #  - normalize=True
    #  - flatten=True
    #  - one_hot_label=False

    # 計算用配列
    x_train_int = Array{Int}(undef, 60000, 784)
    t_train_int = Array{Int}(undef, 60000, 1)
    x_test_int = Array{Int}(undef, 10000, 784)
    t_test_int = Array{Int}(undef, 10000, 1)
        
    # シリアライズドファイル名
    serializedFileName = "mnistDataSet/mnist_data.dat"
    
    
    # シリアライズドファイルの確認
    # シリアライズドファイルが存在する場合は、そのファイルを読み込む。
    if filesize(serializedFileName) != 0
        
        data = deserialize(serializedFileName)
        
        x_train_int = data["x_train_int"]
        t_train_int = data["t_train_int"]
        x_test_int = data["x_test_int"]
        t_test_int = data["t_test_int"]

        
    else
        # 無い場合は、MNISTデータファイルを読み込む

        # 読み込み対象ファイル
        file1 = "mnistDataSet/train-images-idx3-ubyte"
        file2 = "mnistDataSet/train-labels-idx1-ubyte"
        file3 = "mnistDataSet/t10k-images-idx3-ubyte"
        file4 = "mnistDataSet/t10k-labels-idx1-ubyte"
        
        
        # file1について読み込み
     
        fio1 = open(file1, "r")
        dum = read(fio1, 4)
        dum = read(fio1, 4)
        dum = read(fio1, 4)
        dum = read(fio1, 4)
    
        #    while !eof(fio1)
            for r = 1:60000
                for c = 1:784
                    x_train_int[r, c] = read(fio1, UInt8)
                end
            end
        #    end
        

        # file2について読み込み

        fio2 = open(file2, "r")
        dum = read(fio2, 4)
        dum = read(fio2, 4)

        #    while !eof(fio2)
            for r = 1:60000
                t_train_int[r, 1] = read(fio2, UInt8)
            end
        #    end
    
        
        # file3について読み込み
    
        fio3 = open(file3, "r")
        dum = read(fio3, 4)
        dum = read(fio3, 4)
        dum = read(fio3, 4)
        dum = read(fio3, 4)
    
        #    while !eof(fio3)
            for r = 1:10000
                for c = 1:784
                    x_test_int[r, c] = read(fio3, UInt8)
                end
            end
        #    end
    
        
        # file4について読み込み

        fio4 = open(file4, "r")
        dum = read(fio4, 4)
        dum = read(fio4, 4)

        #    while !eof(fio2)
            for r = 1:10000
                t_test_int[r, 1] = read(fio4, UInt8)
            end
        #    end
        
        
        # シリアライズ化(シリアライズドファイルに保存)
        data = Dict()
        data["x_train_int"] = x_train_int
        data["t_train_int"] = t_train_int
        data["x_test_int"] = x_test_int
        data["t_test_int"] = t_test_int
        
        serialize(serializedFileName, data)
        
    end

    
    # データ加工
    
    # normalizeに関する加工
    x_train_flo = Array{Float64}(undef, 60000, 784)
    x_test_flo = Array{Float64}(undef, 10000, 784)

    if normalize
        for r = 1:60000
            for c = 1:784
                x_train_flo[r, c] = x_train_int[r, c] / 255.0
            end
        end
        for r = 1:10000
            for c = 1:784
                x_test_flo[r, c] = x_test_int[r, c] / 255.0
            end
        end
    end
    
    
    #flattenに関する加工
    x_train_int_pixel_value = Matrix{Int}(undef, 28, 28)
    x_test_int_pixel_value = Array{Int}(undef, 1, 28, 28)
    x_train_flo_pixel_value = Matrix{Float64}(undef, 28, 28)
    x_test_flo_pixel_value = Array{Float64}(undef, 1, 28, 28)
    x_train_nonflatten = Matrix{Array}(undef, 60000, 1)
    x_test_nonflatten = Array{Array}(undef, 10000, 1)

    if flatten == false
        
        if normalize
            for i = 1:60000
                j = 1
                for r = 1:28
                    for c = 1:28
                        x_train_flo_pixel_value[r, c] = x_train_flo[i, j]
                        j += 1
                    end
                end
                x_train_nonflatten[i, 1] = x_train_flo_pixel_value
            end
            for i = 1:10000
                j = 1
                for r = 1:28
                    for c = 1:28
                        x_test_flo_pixel_value[1, r, c] = x_test_flo[i, j]
                        j += 1
                    end
                end
                x_test_nonflatten[i, 1] = x_test_flo_pixel_value
            end
        else
            for i = 1:60000
                j = 1
                for r = 1:28
                    for c = 1:28
                        x_train_int_pixel_value[r, c] = x_train_int[i, j]
                        j += 1
                    end
                end
                x_train_nonflatten[i, 1] = x_train_int_pixel_value
            end
            for i = 1:10000
                j = 1
                for r = 1:28
                    for c = 1:28
                        x_test_int_pixel_value[1, r, c] = x_test_int[i, j]
                        j += 1
                    end
                end
                x_test_nonflatten[i, 1] = x_test_int_pixel_value
            end
        end
        
    end
    
    
    # 返却処理
    
    # dmpファイルへの書き出し
    #  存在しない場合のみ、書き出し
    
    
    
    if normalize
        if flatten
            return x_train_flo, change_one_hot(t_train_int, one_hot_label), x_test_flo, change_one_hot(t_test_int, one_hot_label)
        else
            return x_train_nonflatten, change_one_hot(t_train_int, one_hot_label), x_test_nonflatten, change_one_hot(t_test_int, one_hot_label)
        end
    else
        if flatten
            return x_train_int, change_one_hot(t_train_int, one_hot_label), x_test_int, change_one_hot(t_test_int, one_hot_label)
        else
            return x_train_nonflatten, change_one_hot(t_train_int, one_hot_label), x_test_nonflatten, change_one_hot(t_test_int, one_hot_label)
        end
    end
    
    
end

function change_one_hot(arry, one_hot_flg)
    
    number_sample = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    # 配列数を取得
    array_nums = length(arry)
    # 返却用配列
    return_array = Array{Matrix}(undef, array_nums, 1)
    
    if one_hot_flg
        
        for r = 1:array_nums
            one_hot_array = [0 0 0 0 0 0 0 0 0 0]
            for idx = 1:10
                if arry[r, 1] == number_sample[idx]
                    one_hot_array[idx] = 1
                    break
                end
            end
            return_array[r, 1] = one_hot_array            
        end
        
        return return_array
        
    else
        return arry
    end
    
end
