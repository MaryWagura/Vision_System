MODULE MainModule
	TASK PERS tooldata tool1:=[TRUE,[[-92.8336,0.348938,179.712],[1,0,0,0]],[0.5,[50,0,50],[1,0,0,0],0,0,0]];
	TASK PERS wobjdata wobj1:=[FALSE,TRUE,"",[[471.363,-36.723,-4.04044],[0.99971,0.000340047,-0.00208883,0.0239866]],[[0,0,0],[1,0,0,0]]];
	CONST robtarget p10:=[[32.35,-134.49,301.86],[0.0702306,0.0107216,-0.997241,-0.0214989],[-1,-1,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
	CONST robtarget p20:=[[264.56,-154.21,11.15],[0.0702991,0.010664,-0.997237,-0.0214902],[-1,-1,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
	CONST robtarget p30:=[[205.60,-218.52,11.68],[0.07035,0.0106954,-0.997232,-0.0215566],[-1,-1,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
	CONST robtarget p40:=[[258.49,-281.88,10.53],[0.0703828,0.0106784,-0.997229,-0.0215889],[-1,-1,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
	CONST robtarget aruco:=[[118.64,193.32,-5.63],[0.0702491,0.0107088,-0.997241,-0.0214741],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
	 ! --- Heights (mm, wobj1) ---
    CONST num PAPER_Z := 0;         ! measured: cup sucked onto the paper
    CONST num SHAPE_THICK := 10;    ! all shapes are 1 cm thick
    CONST num SQUEEZE := 0;         ! already included in PAPER_Z measurement
    CONST num MIN_ABOVE_PAPER := 2;           ! crash guard: never closer to the paper
    CONST num PLACE_DZ := 0;        ! extra release height above p20-p40

    ! --- Allowed pick area, relative to the marker centre (mm) ---
    CONST num REACH_X := 200;
    CONST num REACH_Y := 140;

    ! --- Slot bookkeeping (how many shapes are already in each slot) ---
    ! PERS: survives a program restart. Reset with option 9 after emptying the board.
    PERS num nCircle := 0;
    PERS num nStar := 1;
    PERS num nSquare := 1;
    CONST num MAX_PER_SLOT := 3;    ! stop before a slot overflows

    PROC PickAndPlace(robtarget pickTarget, robtarget placeTarget)
        MoveJ p10, v1000, fine, tool1\WObj:=wobj1;

        TPWrite "Moving to pick object";
        MoveL pickTarget, v500, fine, tool1\WObj:=wobj1;

        TPWrite "Vacuum ON";
        Set doValve1;
        WaitTime 0.5;

        MoveJ p10, v1000, fine, tool1\WObj:=wobj1;

        TPWrite "Moving to placing position";
        MoveL placeTarget, v500, fine, tool1\WObj:=wobj1;

        TPWrite "Vacuum OFF";
        Reset doValve1;
        WaitTime 1;

        MoveJ p10, v1000, fine, tool1\WObj:=wobj1;
    ENDPROC

    ! Split a comma-separated reply from the Pi ("blue,red") into items{1..n}.
    PROC SplitList(string listStr, INOUT string items{*}, INOUT num n)
        VAR num startPos;
        VAR num commaPos;

        n := 0;
        startPos := 1;
        WHILE startPos <= StrLen(listStr) AND n < Dim(items, 1) DO
            commaPos := StrFind(listStr, startPos, ",");
            n := n + 1;
            items{n} := StrPart(listStr, startPos, commaPos - startPos);
            startPos := commaPos + 1;
        ENDWHILE
    ENDPROC

    ! Show a numbered menu built from what the camera sees and let the operator
    ! choose with TPReadNum. Returns the chosen item, "refresh" (0),
    ! "reset" (9, only when allowReset) or "" for an invalid number.
    FUNC string ChooseFrom(string question, string listStr, bool allowReset)
        VAR string items{5};
        VAR num n;
        VAR num i;
        VAR num choice;

        SplitList listStr, items, n;

        TPWrite question;
        FOR i FROM 1 TO n DO
            TPWrite NumToStr(i, 0) + " = " + items{i};
        ENDFOR
        TPWrite "0 = Refresh the list";
        IF allowReset THEN
            TPWrite "9 = Board emptied (reset slot counters)";
        ENDIF
        TPReadNum choice, "Enter a number:";

        IF choice >= 1 AND choice <= n AND choice = Round(choice) THEN
            RETURN items{choice};
        ELSEIF choice = 0 THEN
            RETURN "refresh";
        ELSEIF choice = 9 AND allowReset THEN
            RETURN "reset";
        ENDIF
        RETURN "";
    ENDFUNC

    PROC main()
        VAR string colour;
        VAR string shape;
        VAR string colourList;
        VAR string shapeList;
        VAR string visionData;
        VAR num comma1;
        VAR num comma2;
        VAR num comma3;
        VAR string str_shape;
        VAR string str_x;
        VAR string str_y;
        VAR string str_angle;
        VAR num val_x;
        VAR num val_y;
        VAR num val_angle;
        VAR bool ok_x;
        VAR bool ok_y;
        VAR bool ok_angle;
        VAR num pickZ;
        VAR num stackN;
        VAR num nCycles;
        VAR bool known_shape;
        VAR robtarget dynamicPickTarget;
        VAR robtarget placeBase;
        VAR robtarget placeTarget;

        TPWrite "Vision Pick and Place (Grade D) Started";

        RobotAsClientConnect;
        nCycles := 0;

        ! ---- Endless picking cycles: one user-selected object per cycle ----
        ! The menus only offer colours/shapes the camera can see right now.
        WHILE TRUE DO
            TPErase;
            RobotClienSendMessage("LIST_COLORS");
            colourList := RobotClientReciveMessage();

            IF colourList = "" OR colourList = "NONE" OR colourList = "UNKNOWN_COMMAND" THEN
                TPWrite "No pickable shapes on the paper - checking again...";
                WaitTime 2;
                colour := "refresh";
            ELSE
                colour := ChooseFrom("Which colour should the robot pick?", colourList, TRUE);
            ENDIF

            IF colour = "refresh" THEN
                ! Nothing chosen: loop and ask the camera again
                WaitTime 0.1;

            ELSEIF colour = "reset" THEN
                nCircle := 0;
                nStar := 0;
                nSquare := 0;
                TPWrite "Slot counters reset - board is empty.";
                WaitTime 1;

            ELSEIF colour = "" THEN
                TPWrite "Invalid choice - please enter a number from the list.";
                WaitTime 2;

            ELSE
                RobotClienSendMessage("LIST_SHAPES," + colour);
                shapeList := RobotClientReciveMessage();

                IF shapeList = "" OR shapeList = "NONE" OR shapeList = "UNKNOWN_COMMAND" THEN
                    TPWrite "No " + colour + " shapes left on the paper.";
                    WaitTime 2;
                    shape := "refresh";
                ELSE
                    shape := ChooseFrom("Which " + colour + " shape should the robot pick?", shapeList, FALSE);
                ENDIF

                IF shape = "refresh" THEN
                    ! Back to the colour menu
                    WaitTime 0.1;

                ELSEIF shape = "" THEN
                    TPWrite "Invalid choice - please enter a number from the list.";
                    WaitTime 2;
                ELSE
                    TPWrite "Looking for a " + colour + " " + shape + "...";
                    RobotClienSendMessage("REQUEST_COORDS," + colour + "," + shape);
                    visionData := RobotClientReciveMessage();

                    IF visionData = "" OR visionData = "NO_TARGET,0,0,0" THEN
                        TPWrite "No " + colour + " " + shape + " found on the paper.";
                        WaitTime 2;
                    ELSE
                        comma1 := StrFind(visionData, 1, ",");

                        IF comma1 > 0 AND comma1 < StrLen(visionData) THEN
                            comma2 := StrFind(visionData, comma1 + 1, ",");
                        ELSE
                            comma2 := 0;
                        ENDIF

                        IF comma2 > 0 AND comma2 < StrLen(visionData) THEN
                            comma3 := StrFind(visionData, comma2 + 1, ",");
                        ELSE
                            comma3 := 0;
                        ENDIF

                        IF comma1 > 0 AND comma2 > 0 AND comma3 > 0 THEN
                            str_shape := StrPart(visionData, 1, comma1 - 1);
                            str_x := StrPart(visionData, comma1 + 1, comma2 - comma1 - 1);
                            str_y := StrPart(visionData, comma2 + 1, comma3 - comma2 - 1);
                            str_angle := StrPart(visionData, comma3 + 1, StrLen(visionData) - comma3);

                            ok_x := StrToVal(str_x, val_x);
                            ok_y := StrToVal(str_y, val_y);
                            ok_angle := StrToVal(str_angle, val_angle);
                            IF ok_angle = FALSE THEN
                                val_angle := 0;
                            ENDIF

                            ! Slot for this shape + how many are already in it
                            known_shape := TRUE;
                            IF str_shape = "circle" THEN
                                placeBase := p20;
                                stackN := nCircle;
                            ELSEIF str_shape = "star" THEN
                                placeBase := p30;
                                stackN := nStar;
                            ELSEIF str_shape = "square" THEN
                                placeBase := p40;
                                stackN := nSquare;
                            ELSE
                                known_shape := FALSE;
                                stackN := 0;
                            ENDIF

                            ! Pick height: top face of a 1 cm shape
                            pickZ := aruco.trans.z + SHAPE_THICK - SQUEEZE;
                            IF pickZ <  aruco.trans.z + MIN_ABOVE_PAPER THEN
                                pickZ :=  aruco.trans.z + MIN_ABOVE_PAPER;
                            ENDIF

                            IF ok_x = FALSE OR ok_y = FALSE THEN
                                TPWrite "Bad coordinates received: " + visionData;
                                WaitTime 2;
                            ELSEIF known_shape = FALSE OR str_shape <> shape THEN
                                TPWrite "Unexpected answer - skipped: " + str_shape;
                                WaitTime 2;
                            ELSEIF Abs(val_x - aruco.trans.x) > REACH_X OR Abs(val_y - aruco.trans.y) > REACH_Y THEN
                                TPWrite "Target outside pick area - skipped";
                                WaitTime 2;
                            ELSEIF stackN >= MAX_PER_SLOT THEN
                                TPWrite "The " + str_shape + " slot is full - empty it, then choose 9.";
                                WaitTime 2;
                            ELSE
                                dynamicPickTarget := aruco;
                                dynamicPickTarget.trans := [val_x, val_y, pickZ];
                                dynamicPickTarget := RelTool(dynamicPickTarget, 0, 0, 0 \Rz:=val_angle);

                                ! Release each extra shape one thickness higher
                                placeTarget := Offs(placeBase, 0, 0, PLACE_DZ + stackN * SHAPE_THICK);

                                TPWrite "Press yellow button to pick the " + colour + " " + shape;
                                WaitDI diOkButton, 1;

                                PickAndPlace dynamicPickTarget, placeTarget;

                                IF str_shape = "circle" THEN
                                    nCircle := nCircle + 1;
                                ELSEIF str_shape = "star" THEN
                                    nStar := nStar + 1;
                                ELSE
                                    nSquare := nSquare + 1;
                                ENDIF
                                nCycles := nCycles + 1;
                                TPWrite "Cycle " + NumToStr(nCycles, 0) + " done: placed the " + colour + " " + shape + ".";
                                WaitTime 1;
                            ENDIF
                        ELSE
                            TPWrite "Bad data received: " + visionData;
                            WaitTime 2;
                        ENDIF
                    ENDIF
                ENDIF
            ENDIF
        ENDWHILE
    ENDPROC
ENDMODULE
