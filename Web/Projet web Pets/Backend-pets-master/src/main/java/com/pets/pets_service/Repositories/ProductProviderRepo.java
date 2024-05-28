package com.pets.pets_service.Repositories;

import com.pets.pets_service.Models.ProductProvider;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.transaction.annotation.Transactional;
import java.util.Optional;

@CrossOrigin("http://localhost:4200/")
@Repository
public interface ProductProviderRepo extends JpaRepository<ProductProvider,Integer> {

    Optional<ProductProvider> findByEmail(String email);

    @Transactional
    @Modifying
    @Query("UPDATE ProductProvider a " +
            "SET a.enabled = TRUE WHERE a.email = ?1")
    int enablePP(String email);
}
