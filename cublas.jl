libcublas=dlopen("libcublas")

# Matrix multiply
for (fname, elty) in ((:cublasSgemm, :Float32), 
                      (:cublasDgemm, :Float64),
                      (:cublasCgemm, :Complex64), 
                      (:cublasZgemm, :Complex128))
    @eval begin
        function cuda_gemm(transA::Char, transB::Char,
                           m::Int32, n::Int32, k::Int32,
                           alpha::($elty), A::Ptr{$elty}, lda::Int32,
                           B::Ptr{$elty}, ldb::Int32, beta::($elty),
                           C::Ptr{$elty}, ldc::Int32)
            ccall(dlsym(libcublas, $string(fname)),
                  Void, (Char, Char, Int32, Int32, Int32,
                         $elty, Ptr{$elty}, Int32, Ptr{$elty}, Int32,
                         $elty, Ptr{$elty}, Int32),
                  transA, transB, m, n, k, alpha, A, lda, B, ldb,
                  beta, C, ldc)
        end
    end
end

# Find the index of the absolute maximum value
for (fname, elty) in ((:cublasIdamax, :Float64), 
                      (:cublasIsamax, :Float32),
                      (:cublasIcamax, :Complex64), 
                      (:cublasIzamax, :Complex128))
    @eval begin
        function cuda_amax(num::Int32, x::Ptr{$elty})
            ccall(dlsym(libcublas, $string(fname)),
                  Int32, (Int32, Ptr{$elty}, Int32),
                  num, x, 1)
        end
    end
end

# Find the index of the absolute minimum value
for (fname, elty) in ((:cublasIdamin, :Float64), 
                      (:cublasIsamin, :Float32),
                      (:cublasIcamin, :Complex64), 
                      (:cublasIzamin, :Complex128))
    @eval begin
        function cuda_amin(num::Int32, x::Ptr{$elty})
            ccall(dlsym(libcublas, $string(fname)),
                  Int32, (Int32, Ptr{$elty}, Int32),
                  num, x, 1)
        end
    end
end

# Find the sum of absolute values
for (fname, elty) in ((:cublasDasum, :Float64), 
                      (:cublasSasum, :Float32),
                      (:cublasScasum, :Complex64), 
                      (:cublasDzasum, :Complex128))
    @eval begin
        function cuda_asum(num::Int32, x::Ptr{$elty})
            ccall(dlsym(libcublas, $string(fname)),
                  $elty, (Int32, Ptr{$elty}, Int32),
                  num, x, 1)
        end
    end
end

# Scale the matrix
for (fname, elty1, elty2) in ((:cublasSscal, :Float32, :Float32), 
                              (:cublasDscal, :Float64, :Float64),
                              (:cublasCscal, :Complex64, :Complex64),
                              (:cublasCsscal, :Float32, :Complex64),
                              (:cublasZscal, :Complex128, :Complex128),
                              (:cublasZdscal, :Float64, :Complex128))
    @eval begin
        function cuda_scal(num::Int32, x::Ptr{$elty2}, alpha::($elty1))
            ccall(dlsym(libcublas, $string(fname)),
                  Void, (Int32, $elty1, Ptr{$elty2}, Int32),
                  num, alpha, x, 1)
        end
    end
end

# Dot product
for (fname, elty) in ((:cublasSdot, :Float32),
                      (:cublasDdot, :Float64),
                      (:cublasCdotu, :Complex64),
                      (:cublasCdotc, :Complex64),
                      (:cublasZdotu, :Complex128),
                      (:cublasZdotc, :Complex128))
    @eval begin
        function cuda_dot(num::Int32, x::Ptr{$elty}, y::Ptr{$elty})
            ccall(dlsym(libcublas, $string(fname)),
                  ($elty) , (Int32, Ptr{$elty}, Int32, Ptr{$elty}, Int32),
                  num, x, 1, y, 1)
        end
    end
end

for (fname, elty) in ((:cublasSaxpy, :Float32),
                      (:cublasDaxpy, :Float64),
                      (:cublasCaxpy, :Complex64),
                      (:cublasZaxpy, :Complex128))
    @eval begin
        function cuda_axpy(n::Int32, alpha::($elty), x::Ptr{$elty}, y::Ptr{$elty})
            ccall(dlsym(libcublas, $string(fname)),
                  Void, (Int32, $elty, Ptr{$elty}, Int32, 
                         Ptr{$elty}, Int32),
                  n, alpha, x, 1, y, 1)
        end
    end
end

# copy matrix
for (fname, elty) in ((:cublasScopy, :Float32),
                      (:cublasDcopy, :Float64),
                      (:cublasCcopy, :Complex64),
                      (:cublasZcopy, :Complex128))
    @eval begin
        function cuda_copy(n::Int32, x::Ptr{$elty}, y::Ptr{$elty})
            ccall(dlsym(libcublas, $string(fname)),
                  Void, (Int32, Ptr{$elty}, Int32, 
                         Ptr{$elty}, Int32),
                  n, x, 1, y, 1)
        end
    end
end

# Norm
for (fname, elty) in ((:cublasSnrm2, :Float32),
                      (:cublasDnrm2, :Float64),
                      (:cublasScnrm2, :Complex64),
                      (:cublasDznrm2, :Complex128))
    @eval begin
        function cuda_nrm2(n::Int32, x::Ptr{$elty})
            ccall(dlsym(libcublas, $string(fname)),
                  ($elty), (Int32, Ptr{$elty}, Int32),
                  n, x, 1)
        end
    end
end

# Swap
for (fname, elty) in ((:cublasSswap, :Float32),
                      (:cublasDswap, :Float64),
                      (:cublasCswap, :Complex64),
                      (:cublasZswap, :Complex128))
    @eval begin
        function cuda_swap(n::Int32, x::Ptr{$elty}, y::Ptr{$elty})
            ccall(dlsym(libcublas, $string(fname)),
                  Void, (Int32, Ptr{$elty}, Int32, Ptr{$elty}, Int32, Ptr{$elty}, Int32),
                  n, x, 1, y, 1)
        end
    end
end

# Rotate
for (fname, elty1, elty2, elty3) in ((:cublasSrot, :Float32, :Float32, :Float32),
                                     (:cublasDrot, :Float64, :Float64, :Float64),
                                     (:cublasCrot, :Complex64, :Float32, :Complex64),
                                     (:cublasZrot, :Complex128, :Float64, :Complex128),
                                     (:cublasCsrot, :Complex64, :Float32, :Float32),
                                     (:cublasZdrot, :Complex128, :Float64, :Float64))
    @eval begin
        function cuda_rot(n::Int32, x::Ptr{$elty1}, y::Ptr{$elty1}, sc::($elty2), ss::($elty3))
            ccall(dlsym(libcublas, $string(fname)),
                  Void, (Int32, Ptr{$elty1}, Int32, Ptr{$elty1}, Int32, $elty2, $elty3),
                  n, x, 1, y, 1, sc, ss)
        end
    end
end

# rotg
for (fname, elty1, elty2) in ((:cublasSrotg, :Float32, :Float32),
                              (:cublasDrotg, :Float64, :Float64),
                              (:cublasCrotg, :Complex64, :Float32),
                              (:cublasZrotg, :Complex128, :Float64))
    @eval begin
        function cuda_rotg(sa::Ptr{$elty1}, sb::Ptr{$elty1}, sc::Ptr{$elty2}, ss::Ptr{$elty1})
            ccall(dlsym(libcublas, $string(fname)),
                  Void, (Ptr{$elty1}, Ptr{$elty1}, Ptr{$elty2}, Ptr{$elty1}),
                  sa, sb, sc, ss)
        end
    end
end

# rotm
for (fname, elty) in ((:cublasSrotm, :Float32),
                      (:cublasDrotm, :Float64))
    @eval begin
        function cuda_rotm(n::Int32, x::Ptr{$elty}, y::Ptr{$elty}, sparam::Ptr{$elty})
            ccall(dlsym(libcublas, $string(fname)),
                  Void, (Int32, Ptr{$elty}, Int32, Ptr{$elty}, Int32, Ptr{$elty}),
                  n, x, 1, y, 1, sparam)
        end
    end
end

# rotmg
for (fname, elty) in ((:cublasSrotmg, :Float32),
                      (:cublasDrotmg, :Float64))
    @eval begin
        function cuda_rotmg(sd1::Ptr{$elty}, sd2::Ptr{$elty}, sx1::Ptr{$elty}, 
                            sy1::Ptr{$elty}, sparam::Ptr{$elty})
            ccall(dlsym(libcublas, $string(fname)),
                  Void, (Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ptr{$elty}),
                  sd1, sd2, sx1, sy1, sparam)
        end
    end
end

# gemv
for (fname, elty) in ((:cublasSgemv, :Float32),
                      (:cublasDgemv, :Float64),
                      (:cublasCgemv, :Complex64),
                      (:cublasZgemv, :Complex128))
    @eval begin
        function cuda_gemv(trans::Char, m::Int32, n::Int32, alpha::($elty), A::Ptr{$elty},
                           lda::Int32, x::Ptr{$elty}, beta::($elty), y::Ptr{$elty})
            ccall(dlsym(libcublas, $string(fname)),
                  Void, (Char, Int32, Int32, $elty, Ptr{$elty}, Int32, 
                         Ptr{$elty}, Int32, $elty, Ptr{$elty}, Int32),
                  trans, m, n, alpha, A, lda, x, 1, beta, y, 1)
        end
    end
end

# gbmv
for (fname, elty) in ((:cublasSgbmv, :Float32),
                      (:cublasDgbmv, :Float64),
                      (:cublasCgbmv, :Complex64),
                      (:cublasZgbmv, :Complex128))
    @eval begin
        function cuda_gbmv(trans::Char, m::Int32, n::Int32, kl::Int32, ku::Int32, alpha::($elty), 
                           A::Ptr{$elty}, lda::Int32, x::Ptr{$elty}, beta::($elty), y::Ptr{$elty})
            ccall(dlsym(libcublas, $string(fname)),
                  Void, (Char, Int32, Int32, Int32, Int32, $elty, Ptr{$elty}, Int32, 
                         Ptr{$elty}, Int32, $elty, Ptr{$elty}, Int32),
                  trans, m, n, kl, ku, alpha, A, lda, x, 1, beta, y, 1)
        end
    end
end

# trmv
for (fname, elty) in ((:cublasStrmv, :Float32),
                      (:cublasDtrmv, :Float64),
                      (:cublasCtrmv, :Complex64),
                      (:cublasZtrmv, :Complex128))
    @eval begin
        function cuda_trmv(uplo::Char, trans::Char, diag::Char, n::Int32, 
                           A::Ptr{$elty}, lda::Int32, x::Ptr{$elty})
            ccall(dlsym(libcublas, $string(fname)),
                  Void, (Char, Char, Char, Int32, Ptr{$elty}, Int32, Ptr{$elty}, Int32),
                  uplo, trans, diag, n, A, lda, x, 1)
        end
    end
end

# tbmv
for (fname, elty) in ((:cublasStbmv, :Float32),
                      (:cublasDtbmv, :Float64),
                      (:cublasCtbmv, :Complex64),
                      (:cublasZtbmv, :Complex128))
    @eval begin
        function cuda_tbmv(uplo::Char, trans::Char, diag::Char, n::Int32, k::Int32,
                           A::Ptr{$elty}, lda::Int32, x::Ptr{$elty})
            ccall(dlsym(libcublas, $string(fname)),
                  Void, (Char, Char, Char, Int32, Int32, Ptr{$elty}, Int32, Ptr{$elty}, Int32),
                  uplo, trans, diag, n, k, A, lda, x, 1)
        end
    end
end

# tpmv
for (fname, elty) in ((:cublasStpmv, :Float32),
                      (:cublasDtpmv, :Float64),
                      (:cublasCtpmv, :Complex64),
                      (:cublasZtpmv, :Complex128))
    @eval begin
        function cuda_tpmv(uplo::Char, trans::Char, diag::Char, n::Int32, 
                           A::Ptr{$elty}, x::Ptr{$elty})
            ccall(dlsym(libcublas, $string(fname)),
                  Void, (Char, Char, Char, Int32, Ptr{$elty}, Ptr{$elty}, Int32),
                  uplo, trans, diag, n, A, x, 1)
        end
    end
end

# trsv
for (fname, elty) in ((:cublasStrsv, :Float32),
                      (:cublasDtrsv, :Float64),
                      (:cublasCtrsv, :Complex64),
                      (:cublasZtrsv, :Complex128))
    @eval begin
        function cuda_trsv(uplo::Char, trans::Char, diag::Char, n::Int32, 
                           A::Ptr{$elty}, lda::Int32, x::Ptr{$elty})
            ccall(dlsym(libcublas, $string(fname)),
                  Void, (Char, Char, Char, Int32, Ptr{$elty}, Int32, Ptr{$elty}, Int32),
                  uplo, trans, diag, n, A, lda, x, 1)
        end
    end
end

# tpsv
for (fname, elty) in ((:cublasStpsv, :Float32),
                      (:cublasDtpsv, :Float64),
                      (:cublasCtpsv, :Complex64),
                      (:cublasZtpsv, :Complex128))
    @eval begin
        function cuda_tpsv(uplo::Char, trans::Char, diag::Char, n::Int32, 
                           A::Ptr{$elty}, x::Ptr{$elty})
            ccall(dlsym(libcublas, $string(fname)),
                  Void, (Char, Char, Char, Int32, Ptr{$elty}, Ptr{$elty}, Int32),
                  uplo, trans, diag, n, A, x, 1)
        end
    end
end

# tbsv
for (fname, elty) in ((:cublasStbsv, :Float32),
                      (:cublasDtbsv, :Float64),
                      (:cublasCtbsv, :Complex64),
                      (:cublasZtbsv, :Complex128))
    @eval begin
        function cuda_tbsv(uplo::Char, trans::Char, diag::Char, n::Int32, k::Int32,
                           A::Ptr{$elty}, lda::Int32, x::Ptr{$elty})
            ccall(dlsym(libcublas, $string(fname)),
                  Void, (Char, Char, Char, Int32, Int32, Ptr{$elty}, Int32, Ptr{$elty}, Int32),
                  uplo, trans, diag, n, k, A, lda, x, 1)
        end
    end
end

# symv
for (fname, elty) in ((:cublasSsymv, :Float32),
                      (:cublasDsymv, :Float64))
    @eval begin
        function cuda_symv(uplo::Char, n::Int32, alpha::($elty), A::Ptr{$elty}, 
                           lda::Int32, x::Ptr{$elty}, beta::($elty), y::Ptr{$elty})
            ccall(dlsym(libcublas, $string(fname)),
                  Void, (Char, Int32, $elty, Ptr{$elty}, Int32, Ptr{$elty}, 
                         Int32, $elty, Ptr{$elty}, Int32),
                  uplo, n, alpha, A, lda, x, 1, beta, y, 1)
        end
    end
end

# hemv
for (fname, elty) in ((:cublasChemv, :Complex64),
                      (:cublasZhemv, :Complex128))
    @eval begin
        function cuda_hemv(uplo::Char, n::Int32, alpha::($elty), A::Ptr{$elty}, 
                           lda::Int32, x::Ptr{$elty}, beta::($elty), y::Ptr{$elty})
            ccall(dlsym(libcublas, $string(fname)),
                  Void, (Char, Int32, $elty, Ptr{$elty}, Int32, Ptr{$elty}, 
                         Int32, $elty, Ptr{$elty}, Int32),
                  uplo, n, alpha, A, lda, x, 1, beta, y, 1)
        end
    end
end

# sbmv
for (fname, elty) in ((:cublasSsbmv, :Float32),
                      (:cublasDsbmv, :Float64))
    @eval begin
        function cuda_sbmv(uplo::Char, n::Int32, k::Int32, alpha::($elty), A::Ptr{$elty}, 
                           lda::Int32, x::Ptr{$elty}, beta::($elty), y::Ptr{$elty})
            ccall(dlsym(libcublas, $string(fname)),
                  Void, (Char, Int32, Int32, $elty, Ptr{$elty}, Int32, Ptr{$elty}, 
                         Int32, $elty, Ptr{$elty}, Int32),
                  uplo, n, k, alpha, A, lda, x, 1, beta, y, 1)
        end
    end
end

# hbmv
for (fname, elty) in ((:cublasChemv, :Complex64),
                      (:cublasZhemv, :Complex128))
    @eval begin
        function cuda_hbmv(uplo::Char, n::Int32, k::Int32, alpha::($elty), A::Ptr{$elty}, 
                           lda::Int32, x::Ptr{$elty}, beta::($elty), y::Ptr{$elty})
            ccall(dlsym(libcublas, $string(fname)),
                  Void, (Char, Int32, Int32, $elty, Ptr{$elty}, Int32, Ptr{$elty}, 
                         Int32, $elty, Ptr{$elty}, Int32),
                  uplo, n, k, alpha, A, lda, x, 1, beta, y, 1)
        end
    end
end

# spmv
for (fname, elty) in ((:cublasSspmv, :Float32),
                      (:cublasDspmv, :Float64))
    @eval begin
        function cuda_spmv(uplo::Char, n::Int32, alpha::($elty), A::Ptr{$elty}, 
                           x::Ptr{$elty}, beta::($elty), y::Ptr{$elty})
            ccall(dlsym(libcublas, $string(fname)),
                  Void, (Char, Int32, $elty, Ptr{$elty}, Ptr{$elty}, 
                         Int32, $elty, Ptr{$elty}, Int32),
                  uplo, n, alpha, A, x, 1, beta, y, 1)
        end
    end
end

# hpmv
for (fname, elty) in ((:cublasChpmv, :Complex64),
                      (:cublasZhpmv, :Complex128))
    @eval begin
        function cuda_hpmv(uplo::Char, n::Int32, alpha::($elty), A::Ptr{$elty}, 
                           x::Ptr{$elty}, beta::($elty), y::Ptr{$elty})
            ccall(dlsym(libcublas, $string(fname)),
                  Void, (Char, Int32, $elty, Ptr{$elty}, Ptr{$elty}, 
                         Int32, $elty, Ptr{$elty}, Int32),
                  uplo, n, alpha, A, x, 1, beta, y, 1)
        end
    end
end

# ger
for (fname, elty) in ((:cublasSger, :Float32),
                      (:cublasDger, :Float64))
    @eval begin
        function cuda_ger(m::Int32, n::Int32, alpha::($elty), x::Ptr{$elty}, 
                          y::Ptr{$elty}, A::Ptr{$elty}, lda::Int32)
            ccall(dlsym(libcublas, $string(fname)),
                  Void, (Int32, Int32, $elty, Ptr{$elty}, Int32, Ptr{$elty},
                         Int32, Ptr{$elty}, Int32),
                  m, n, alpha, x, 1, y, 1,  A, lda)
        end
    end
end

# geru
for (fname, elty) in ((:cublasSgeru, :Complex64),
                      (:cublasDgeru, :Complex128))
    @eval begin
        function cuda_geru(m::Int32, n::Int32, alpha::($elty), x::Ptr{$elty}, 
                          y::Ptr{$elty}, A::Ptr{$elty}, lda::Int32)
            ccall(dlsym(libcublas, $string(fname)),
                  Void, (Int32, Int32, $elty, Ptr{$elty}, Int32, Ptr{$elty},
                         Int32, Ptr{$elty}, Int32),
                  m, n, alpha, x, 1, y, 1,  A, lda)
        end
    end
end

# gerc
for (fname, elty) in ((:cublasSgerc, :Complex64),
                      (:cublasDgerc, :Complex128))
    @eval begin
        function cuda_gerc(m::Int32, n::Int32, alpha::($elty), x::Ptr{$elty}, 
                          y::Ptr{$elty}, A::Ptr{$elty}, lda::Int32)
            ccall(dlsym(libcublas, $string(fname)),
                  Void, (Int32, Int32, $elty, Ptr{$elty}, Int32, Ptr{$elty},
                         Int32, Ptr{$elty}, Int32),
                  m, n, alpha, x, 1, y, 1,  A, lda)
        end
    end
end

# syr
for (fname, elty) in ((:cublasSsyr, :Float32),
                      (:cublasDsyr, :Float64))
    @eval begin
        function cuda_syr(uplo::Char, n::Int32, alpha::($elty), x::Ptr{$elty},
                          A::Ptr{$elty}, lda::Int32)
            ccall(dlsym(libcublas, $string(fname)),
                  Void, (Char, Int32, $elty, Ptr{$elty}, Int32, Ptr{$elty}, Int32),
                  uplo, n, alpha, x, 1, A, lda)
        end
    end
end

# her
for (fname, elty) in ((:cublasSher, :Complex64),
                      (:cublasDher, :Complex128))
    @eval begin
        function cuda_her(uplo::Char, n::Int32, alpha::($elty), x::Ptr{$elty},
                          A::Ptr{$elty}, lda::Int32)
            ccall(dlsym(libcublas, $string(fname)),
                  Void, (Char, Int32, $elty, Ptr{$elty}, Int32, Ptr{$elty}, Int32),
                  uplo, n, alpha, x, 1, A, lda)
        end
    end
end

# spr
for (fname, elty) in ((:cublasSspr, :Float32),
                      (:cublasDspr, :Float64))
    @eval begin
        function cuda_syr(uplo::Char, n::Int32, alpha::($elty), x::Ptr{$elty},
                          A::Ptr{$elty})
            ccall(dlsym(libcublas, $string(fname)),
                  Void, (Char, Int32, $elty, Ptr{$elty}, Int32, Ptr{$elty}, Int32),
                  uplo, n, alpha, x, 1, A)
        end
    end
end

# hpr
for (fname, elty) in ((:cublasShpr, :Complex64),
                      (:cublasDhpr, :Complex128))
    @eval begin
        function cuda_hpr(uplo::Char, n::Int32, alpha::($elty), x::Ptr{$elty},
                          A::Ptr{$elty})
            ccall(dlsym(libcublas, $string(fname)),
                  Void, (Char, Int32, $elty, Ptr{$elty}, Int32, Ptr{$elty}, Int32),
                  uplo, n, alpha, x, 1, A)
        end
    end
end

# syr2
for (fname, elty) in ((:cublasSsyr2, :Float32),
                      (:cublasDsyr2, :Float64))
    @eval begin
        function cuda_syr2(uplo::Char, n::Int32, alpha::($elty), x::Ptr{$elty}, y::Ptr{$elty},
                           A::Ptr{$elty}, lda::Int32)
            ccall(dlsym(libcublas, $string(fname)),
                  Void, (Char, Int32, $elty, Ptr{$elty}, Int32, Ptr{$elty}, Int32, Ptr{$elty}, Int32),
                  uplo, n, alpha, x, 1, y, 1, A, lda)
        end
    end
end

# her2
for (fname, elty) in ((:cublasSher2, :Complex64),
                      (:cublasDher2, :Complex128))
    @eval begin
        function cuda_her2(uplo::Char, n::Int32, alpha::($elty), x::Ptr{$elty}, y::Ptr{$elty},
                           A::Ptr{$elty}, lda::Int32)
            ccall(dlsym(libcublas, $string(fname)),
                  Void, (Char, Int32, $elty, Ptr{$elty}, Int32, Ptr{$elty}, Int32, Ptr{$elty}, Int32),
                  uplo, n, alpha, x, 1, y, 1, A, lda)
        end
    end
end

# spr2
for (fname, elty) in ((:cublasSspr2, :Float32),
                      (:cublasDspr2, :Float64))
    @eval begin
        function cuda_spr2(uplo::Char, n::Int32, alpha::($elty), x::Ptr{$elty},
                           y::Ptr{$elty}, A::Ptr{$elty})
            ccall(dlsym(libcublas, $string(fname)),
                  Void, (Char, Int32, $elty, Ptr{$elty}, Int32, y::Ptr{$elty}, Int32, Ptr{$elty}, Int32),
                  uplo, n, alpha, x, 1, y, 1, A)
        end
    end
end

# hpr2
for (fname, elty) in ((:cublasShpr2, :Complex64),
                      (:cublasDhpr2, :Complex128))
    @eval begin
        function cuda_hpr2(uplo::Char, n::Int32, alpha::($elty), x::Ptr{$elty},
                           y::Ptr{$elty}, A::Ptr{$elty})
            ccall(dlsym(libcublas, $string(fname)),
                  Void, (Char, Int32, $elty, Ptr{$elty}, Int32, y::Ptr{$elty}, Int32, Ptr{$elty}, Int32),
                  uplo, n, alpha, x, 1, y, 1, A)
        end
    end
end
