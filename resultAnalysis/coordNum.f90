
program main
    character(len=256) :: fl_in
    character(len=256) :: fl_out
    parameter(nat=12)
    character*2 atom_symb

    ! fl_name = "isolated.ascii"
    ! fl_name = "polymeric.ascii"
    ! fl_name = "tmp.ascii"
    call coordNum(fl_in, fl_out, nat, atom_symb)

end program main

subroutine coordNum(fl_in, fl_out, nat, atom_symb)
       implicit real*8 (a-h,o-z)
       ! parameter(nat=24)
       parameter(rfirst=1.8d0, rsecond=2.25d0)
       parameter(nwork=1000)
       character*2 symb
       character*2 atom_symb
       logical debug
       dimension rxyz(3,nat),alat(3,3),symb(nat),coord(nat),alatalat(3,3)
       dimension eigalat(3),work(nwork)
       character(len=256) :: fl_in
       character(len=256) :: fl_out

       ! real :: cc, ss
       logical :: exist


        debug=.false.
       !open (unit=11,file='polymeric.ascii')
       ! open (unit=11,file='isolated.ascii')
       open (unit=11,file=fl_in)


       read(11,*) natp
        if (debug) write(*,*) natp
        if (natp.gt.nat) stop 'nat'
       read(11,*) dxx,dyx,dyy
        if (debug) write(*,*)  dxx,dyx,dyy
       read(11,*) dzx,dzy,dzz
        if (debug) write(*,*)  dzx,dzy,dzz

        alat(1,1)=dxx
        alat(2,1)=0.d0
        alat(3,1)=0.d0

        alat(1,2)=dyx
        alat(2,2)=dyy
        alat(3,2)=0.d0

        alat(1,3)=dzx
        alat(2,3)=dzy
        alat(3,3)=dzz

      do iat = 1, nat
        read(11, *) ( rxyz(j, iat), j = 1, 3) ,symb(iat)
        if (debug) write(*,'(3(2x,e12.5),4x,a,3x,i4)')  ( rxyz(j, iat), j = 1, 3) ,symb(iat), iat
      end do

      do i=1,3
      do j=1,3
          alatalat(i,j)=alat(1,i)*alat(1,j)+alat(2,i)*alat(2,j)+alat(3,i)*alat(3,j)
      enddo
      enddo
      call dsyev('N', 'L', 3, alatalat, 3, eigalat, work, nwork, info)
      !  write(*,*) 'alat EVals',eigalat
      !  write(*,*) 'ixyzmax',int(sqrt(1.d0/eigalat(1))*rsecond)
      ixyzmax= int(sqrt(1.d0/eigalat(1))*rsecond) + 1
    ! write(*,*) 'ixyzmax ',ixyzmax

   do lat = 1, nat
   coord(lat)=0.d0
   if (symb(lat).eq.atom_symb) then   ! calculate om;y ye coordddintion number between Al and H
     do jat = 1, nat
     if (symb(jat).eq."H") then   ! calculate om;y ye coordddintion number between Al and H
         do ix = -ixyzmax,ixyzmax
           do iy = -ixyzmax,ixyzmax
             do iz = -ixyzmax,ixyzmax
                xj = rxyz(1, jat) + ix*alat(1,1)+iy*alat(1,2)+iz*alat(1,3)
                yj = rxyz(2, jat) + ix*alat(2,1)+iy*alat(2,2)+iz*alat(2,3)
                zj = rxyz(3, jat) + ix*alat(3,1)+iy*alat(3,2)+iz*alat(3,3)
                dist2 = (xj-rxyz(1, lat))**2+(yj-rxyz(2, lat))**2+(zj-rxyz(3, lat))**2
                !write(*,*) xj,rxyz(1, lat),yj,rxyz(2, lat),zj,rxyz(3,lat)

! coordination number calculated with soft cutoff between first
! nearest neighbor distance (rfirst) and midpoint of first and second nearest neighbor (rsecond)
             if (dist2.le.rfirst**2) then
             coord(lat)=coord(lat)+1.d0
             else if (dist2.ge.rsecond**2) then
             else
             rij=sqrt(dist2)
             xarg=(rij-rfirst)*(1.d0/(rsecond-rfirst))
             coord(lat)=coord(lat)+(2*xarg+1.d0)*(xarg-1.d0)**2
             endif
             enddo
           enddo
         enddo
     endif
     enddo
   endif
   enddo


   ! write acii format
    !    open (unit=22,file='OUT.ascii')
    !    write(22,*) nat

    !     write(22,*) dxx,dyx,dyy
    !     write(22,*) dzx,dzy,dzz

    !     dxx=alat(1,1)

    !     dyx=alat(1,2)
    !     dyy=alat(2,2)

    !     dzx=alat(1,3)
    !     dzy=alat(2,3)
    !     dzz=alat(3,3)

    !   do iat = 1, nat
    !     write(22, *) ( rxyz(j, iat), j = 1, 3) ,symb(iat),coord(iat)
    !   end do

   ! write extxyz format
        ! open(unit=22, file=fl_out)

       ! append mode
       inquire(file=fl_out, exist=exist)
       if (exist) then
        open(unit=22, file=fl_out, status="old", position="append", action="write")
       else
        open(unit=22, file=fl_out, status="new", action="write")

        endif

       write(22,"(i6)") nat

        write(22,'(a12,9E16.8,a2)',advance='no') '"Lattice="',dxx,0.0,0.0,dyx,dyy,0.0,dzx,dzy,dzz
        write(22,'(a53)') ' Properties=species:S:1:pos:R:3:coordn:R:1 pbc="T T T"'
        ! write(22,*) dxx,dyx,dyy
        ! write(22,*) dzx,dzy,dzz

        dxx=alat(1,1)

        dyx=alat(1,2)
        dyy=alat(2,2)

        dzx=alat(1,3)
        dzy=alat(2,3)
        dzz=alat(3,3)

      do iat = 1, nat
        write(22, *) symb(iat), ( rxyz(j, iat), j = 1, 3), coord(iat)
      end do
      
      close(22)

      ! cc = 0.0
      ! ss = 0.0
      ! do iat = 1, nat
      ! if (coord(iat) /= 0) then
      !   ss = ss + coord(iat)
      !   cc = cc + 1.0
      ! end if
      ! end do
      ! ss = ss / cc

      ! open (unit=23,file='coordnum.txt')
      ! write(23,*) ss
      ! close(23)


      ! end
end subroutine coordNum
