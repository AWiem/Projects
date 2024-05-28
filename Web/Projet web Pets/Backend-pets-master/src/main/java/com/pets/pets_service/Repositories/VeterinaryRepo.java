package com.pets.pets_service.Repositories;

import com.pets.pets_service.Models.Veterinary;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.transaction.annotation.Transactional;
import java.util.Optional;


@CrossOrigin("http://localhost:4200/")
@Repository
public interface VeterinaryRepo  extends JpaRepository<Veterinary,Integer> {
    Optional<Veterinary> findByEmail(String email);

    @Transactional
    @Modifying
    @Query("UPDATE Veterinary a " +
            "SET a.enabled = TRUE WHERE a.email = ?1")
    int enableVet(String email);
}
